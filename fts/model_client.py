"""OpenAI-compatible chat-completions client (Req 0.9, 0.13, 3.3, 3.4, 6.5).

One client drives the hosted baseline (OpenAI), the local ``llama-server``, and
the load test. It:

* reads the API key only from ``os.environ[api_key_env]`` and, when a key is
  required but unset or empty, raises :class:`MissingSecretError` naming the
  variable before any request (callers exit non-zero);
* enforces a per-request timeout (:class:`ModelTimeoutError`);
* retries HTTP 429 and 5xx with exponential backoff, up to ``max_attempts`` in total;
* sends ``temperature`` and ``seed`` and, if the server rejects one of them with a
  400 naming the parameter, drops it once and records it in ``dropped_params``;
* checks and records every request with a :class:`~fts.spend.SpendLedger` when one is given;
* never logs or prints the key.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from fts.spend import SpendLedger, estimate_prompt_tokens

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class ModelClientError(RuntimeError):
    """Base class for client errors."""


class MissingSecretError(ModelClientError):
    def __init__(self, env_var: str) -> None:
        self.env_var = env_var
        super().__init__(f"required environment variable {env_var} is unset or empty")


class ModelTimeoutError(ModelClientError):
    """The request exceeded the timeout."""


class ModelConnectionError(ModelClientError):
    """The server could not be reached."""


class ModelHTTPError(ModelClientError):
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:500]}")


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str  # as returned by the server
    finish_reason: str | None = None
    usage_reported: bool = True


@dataclass
class ClientConfig:
    base_url: str  # e.g. https://api.openai.com/v1 or http://127.0.0.1:8080/v1
    model: str
    api_key_env: str | None = None
    timeout_s: float = 60.0
    seed: int | None = 0
    temperature: float | None = 0.0
    max_tokens_field: str = "max_completion_tokens"
    extra: dict[str, Any] = field(default_factory=dict)
    droppable: tuple[str, ...] = ("temperature", "seed")
    max_attempts: int = 5
    backoff_s: float = 1.0


def read_secret(env_var: str) -> str:
    """The secret's value; MissingSecretError naming the variable when unset or empty."""
    value = os.environ.get(env_var, "")
    if not value.strip():
        raise MissingSecretError(env_var)
    return value


class ModelClient:
    def __init__(
        self,
        config: ClientConfig,
        ledger: SpendLedger | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        ledger_tag: str = "",
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.ledger_tag = ledger_tag
        self._sleep = sleep
        self._key = read_secret(config.api_key_env) if config.api_key_env else None
        self.dropped_params: list[str] = []
        self._http = httpx.Client(
            timeout=config.timeout_s,
            transport=transport,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=64),
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ModelClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def url(self) -> str:
        return self.config.base_url.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        return headers

    def payload(self, messages: list[dict], max_tokens: int) -> dict:
        body: dict[str, Any] = {"model": self.config.model, "messages": messages, self.config.max_tokens_field: max_tokens}
        if self.config.temperature is not None and "temperature" not in self.dropped_params:
            body["temperature"] = self.config.temperature
        if self.config.seed is not None and "seed" not in self.dropped_params:
            body["seed"] = self.config.seed
        for key, value in self.config.extra.items():
            if key not in self.dropped_params:
                body[key] = value
        return body

    def sent_params(self, max_tokens: int = 1) -> list[str]:
        """Names of the optional parameters the client currently sends."""
        return sorted(k for k in self.payload([], max_tokens) if k not in ("model", "messages"))

    def _rejected_param(self, response: httpx.Response) -> str | None:
        if response.status_code != 400:
            return None
        try:
            err = response.json().get("error", {}) or {}
        except ValueError:
            return None
        param = err.get("param")
        if param in self.config.droppable and param not in self.dropped_params:
            return param
        return None

    def complete(self, messages: list[dict], max_tokens: int) -> Completion:
        if self.ledger is not None:
            self.ledger.check(messages, max_tokens)
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._http.post(self.url, headers=self._headers(), json=self.payload(messages, max_tokens))
            except httpx.TimeoutException as exc:
                raise ModelTimeoutError(f"request timed out after {self.config.timeout_s} s") from exc
            except httpx.TransportError as exc:
                raise ModelConnectionError(f"cannot reach {self.url}: {type(exc).__name__}") from exc
            param = self._rejected_param(response)
            if param is not None:
                self.dropped_params.append(param)
                attempt -= 1  # a dropped parameter is a correction, not a retry
                continue
            if response.status_code in RETRYABLE_STATUS and attempt < self.config.max_attempts:
                self._sleep(self._backoff(response, attempt))
                continue
            if response.status_code != 200:
                raise ModelHTTPError(response.status_code, response.text)
            return self._parse(response.json(), messages, max_tokens)

    def _backoff(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return self.config.backoff_s * 2 ** (attempt - 1)

    def _parse(self, data: dict, messages: list[dict], max_tokens: int) -> Completion:
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        usage = data.get("usage") or {}
        reported = "prompt_tokens" in usage and "completion_tokens" in usage
        if reported:
            prompt_tokens, completion_tokens = int(usage["prompt_tokens"]), int(usage["completion_tokens"])
        else:
            # No usage: charge the conservative estimate so the ledger never undercounts.
            prompt_tokens, completion_tokens = estimate_prompt_tokens(messages), max_tokens
        completion = Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=str(data.get("model", "")),
            finish_reason=choice.get("finish_reason"),
            usage_reported=reported,
        )
        if self.ledger is not None:
            self.ledger.record(prompt_tokens, completion_tokens, completion.model, self.ledger_tag)
        return completion


def client_from_config(cfg: dict, ledger: SpendLedger | None = None, **kwargs: Any) -> ModelClient:
    """Build a client from a config mapping such as config/hosted.yaml or config/serve.yaml."""
    return ModelClient(
        ClientConfig(
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key_env=cfg.get("api_key_env") or None,
            timeout_s=float(cfg.get("timeout_s", 60.0)),
            seed=cfg.get("seed", 0),
            temperature=cfg.get("temperature", 0.0),
            max_tokens_field=cfg.get("max_tokens_field", "max_completion_tokens"),
            extra=dict(cfg.get("extra") or {}),
        ),
        ledger=ledger,
        **kwargs,
    )
