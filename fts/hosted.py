"""Hosted-model setup shared by the M0 API proof and the hosted baseline (Req 0.13, 1.7, 3.3, 3.4).

The hosted model is TypeSafe's Jev (config/hosted.yaml). ``open_hosted`` builds
the spend ledger and a :class:`~fts.jev_client.JevClient` that checks and records
every request in it. ``main`` is the API proof (``scripts/hosted_check.py``): one
request with one yes/no question about a made-up request (no Test data), recording
the ``model`` that answered, the answer, the token counts and the cost in
results/hosted_check.json and the "Hosted model and Spend_Cap" section of
docs/ENVIRONMENT.md. A missing key fails before any request with exit status 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fts import envdoc, repro, settings
from fts.jev_client import JevClient, client_from_config
from fts.model_client import MissingSecretError, ModelClientError
from fts.spend import SpendCapExceeded, SpendLedger, prices_from_config

REPO_ROOT = Path(__file__).resolve().parent.parent
HOSTED_YAML = REPO_ROOT / "config" / "hosted.yaml"
SECTION = "Hosted model and Spend_Cap"
CHECK_STATE = {"request": "wake me up at seven am tomorrow"}
CHECK_QUESTIONS = {"asks_alarm": {"type": "noul", "instructions": "Does `request` ask the voice assistant to set an alarm?"}}


def load_config(path: Path | str = HOSTED_YAML) -> dict:
    return settings.load_yaml(path)


def open_ledger(cfg: dict, root: Path = REPO_ROOT) -> SpendLedger:
    return SpendLedger(root / cfg["ledger_path"], cap_usd=float(cfg["spend_cap_usd"]), prices=prices_from_config(cfg))


def open_hosted(cfg: dict, root: Path = REPO_ROOT, tag: str = "", **client_kwargs) -> tuple[JevClient, SpendLedger]:
    """The ledger and a Jev client that checks and records every request in it."""
    ledger = open_ledger(cfg, root)
    return client_from_config(cfg, ledger=ledger, ledger_tag=tag, **client_kwargs), ledger


def render_section(cfg: dict, check: dict, ledger_summary: dict, stamp: repro.Stamp) -> str:
    prices = cfg["prices_usd_per_1m_tokens"]
    lines = [
        f"- Hosted_Model: TypeSafe Jev, model `{cfg['model']}` (pinned version) via `POST {cfg['url']}`; key read from `{cfg['api_key_env']}`",
        f"- API check: the response's `model` field was `{check['response_model']}`; P(yes) for the check question {check['answer']:.2f}",
        f"- Check tokens: {check['input_tokens']} input, {check['output_tokens']} output; cost ${check['cost_usd']:.8f}",
        f"- Prices (USD per 1M tokens, {cfg['prices_source']}, accessed {cfg['prices_accessed']}): input {prices['input']}, output {prices['output']} "
        f"(output is free); the ledger charges input tokens at {ledger_summary['prompt_price_per_mtok']}",
        f"- **Spend_Cap: ${float(cfg['spend_cap_usd']):.2f}**; ledger `{cfg['ledger_path']}` total after the check: "
        f"${ledger_summary['spend_usd']:.8f} over {ledger_summary['requests']} request(s)",
        *envdoc.stamp_lines(stamp),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, env_md: Path = envdoc.ENVIRONMENT_MD, transport=None) -> int:
    cfg = load_config(root / "config" / "hosted.yaml")
    stamp = repro.start_run(seed=0)
    try:
        client, ledger = open_hosted(cfg, root, tag="hosted_check", transport=transport)
    except MissingSecretError as exc:
        print(f"FAIL: {exc}; no request was sent", file=sys.stderr)
        return 2
    with client:
        try:
            response = client.ask(CHECK_STATE, CHECK_QUESTIONS)
        except SpendCapExceeded as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        except ModelClientError as exc:
            print(f"FAIL: hosted request failed: {exc}", file=sys.stderr)
            return 1
    check = {
        "configured_model": cfg["model"],
        "response_model": response.model,
        "question": CHECK_QUESTIONS["asks_alarm"]["instructions"],
        "state": CHECK_STATE,
        "answer": float(response.answers["asks_alarm"]["noul"]),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": ledger.prices.cost(response.input_tokens, response.output_tokens),
    }
    summary = ledger.summary()
    repro.write_json(root / "results" / "hosted_check.json", {"check": check, "ledger": summary}, stamp)
    envdoc.replace_section(SECTION, render_section(cfg, check, summary, stamp), env_md)
    print(f"hosted check OK: model {check['response_model']}, P(yes) {check['answer']:.2f}, {check['input_tokens']} input tokens, ${check['cost_usd']:.8f}")
    return 0
