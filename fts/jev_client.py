"""Client for TypeSafe's Jev decision model (the hosted baseline; Req 0.9, 0.13, 3.3, 3.4).

``POST https://api.typesafe.ai/v1/systemone`` takes a ``state``, a ``model`` and a
map of typed ``questions`` (choice, noul, score) and returns calibrated answers,
the ``model`` version that answered, and ``usage.input_tokens`` /
``usage.output_tokens`` (only input tokens are billed). The client:

* reads the key only from ``os.environ[api_key_env]`` and raises
  :class:`~fts.model_client.MissingSecretError`, naming the variable, before any
  request when it is unset or empty;
* enforces a per-request timeout;
* retries HTTP 429, 529 and 5xx with exponential backoff (``retry-after`` honoured),
  and transient transport errors (dropped connections, timeouts), up to
  ``max_attempts`` in total. A failed attempt that may have reached the server is
  charged to the ledger at its worst case (tag ``<tag>:unconfirmed``), so the
  ledger stays an upper bound on the bill;
* with a :class:`~fts.spend.SpendLedger`, refuses a request whose worst-case cost
  (request body characters / 3 input tokens) could cross the Spend_Cap, and
  records every completed request;
* never logs or prints the key.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from fts.model_client import MissingSecretError, ModelConnectionError, ModelHTTPError, ModelTimeoutError, read_secret
from fts.spend import SpendLedger, estimate_text_tokens

DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})

__all__ = ["JevClient", "JevResponse", "MissingSecretError", "client_from_config"]


@dataclass(frozen=True)
class JevResponse:
    answers: dict
    input_tokens: int
    output_tokens: int
    model: str  # versioned ID that answered, e.g. "jev-1.13.0"


class JevClient:
    def __init__(
        self,
        model: str,
        api_key_env: str = "TYPESAFE_API_KEY",
        url: str = DEFAULT_URL,
        timeout_s: float = 120.0,
        max_attempts: int = 5,
        backoff_s: float = 1.0,
        ledger: SpendLedger | None = None,
        ledger_tag: str = "",
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.url = url
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.ledger = ledger
        self.ledger_tag = ledger_tag
        self._sleep = sleep
        self._key = read_secret(api_key_env)
        self._http = httpx.Client(timeout=timeout_s, transport=transport)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> JevClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def body(self, state: object, questions: dict) -> str:
        return json.dumps({"model": self.model, "state": state, "questions": questions}, ensure_ascii=False)

    def ask(self, state: object, questions: dict) -> JevResponse:
        """One request; answers are keyed by the caller's question IDs."""
        body = self.body(state, questions)
        if self.ledger is not None:
            self.ledger.check_tokens(estimate_text_tokens(body), 0)
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._http.post(self.url, headers=headers, content=body.encode("utf-8"))
            except httpx.TransportError as exc:
                if not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)) and self.ledger is not None:
                    # The request may have reached the server and been billed: charge its worst case.
                    self.ledger.record(estimate_text_tokens(body), 0, self.model, f"{self.ledger_tag}:unconfirmed")
                if attempt < self.max_attempts:
                    self._sleep(self.backoff_s * 2 ** (attempt - 1))
                    continue
                if isinstance(exc, httpx.TimeoutException):
                    raise ModelTimeoutError(f"Jev request timed out after {self.timeout_s} s") from exc
                raise ModelConnectionError(f"cannot reach {self.url}: {type(exc).__name__}") from exc
            if response.status_code in RETRYABLE_STATUS and attempt < self.max_attempts:
                self._sleep(self._backoff(response, attempt))
                continue
            if response.status_code != 200:
                raise ModelHTTPError(response.status_code, response.text)
            return self._parse(response.json())

    def _backoff(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return self.backoff_s * 2 ** (attempt - 1)

    def _parse(self, data: dict) -> JevResponse:
        usage = data.get("usage") or {}
        result = JevResponse(answers=data["answers"], input_tokens=int(usage.get("input_tokens", 0)),
                             output_tokens=int(usage.get("output_tokens", 0)), model=str(data.get("model", "")))
        if self.ledger is not None:
            self.ledger.record(result.input_tokens, result.output_tokens, result.model, self.ledger_tag)
        return result


def client_from_config(cfg: dict, ledger: SpendLedger | None = None, **kwargs) -> JevClient:
    """A client from config/hosted.yaml."""
    return JevClient(model=cfg["model"], api_key_env=cfg["api_key_env"], url=cfg["url"], timeout_s=float(cfg["timeout_s"]), ledger=ledger, **kwargs)
