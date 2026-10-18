"""Hosted-model setup shared by the M0 API proof and the hosted baseline (Req 0.13, 1.7, 3.3, 3.4).

``open_hosted`` builds the spend ledger and the ``ModelClient`` from
config/hosted.yaml. ``main`` is the API proof (``scripts/hosted_check.py``): one
request through the client and the ledger, recording the returned ``model``
field, the parameters the API accepted, the token counts, and the cost, in
results/hosted_check.json and the "Hosted model and Spend_Cap" section of
docs/ENVIRONMENT.md. A missing key fails before any request with exit status 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fts import envdoc, repro, settings
from fts.model_client import ModelClient, ModelClientError, MissingSecretError, client_from_config
from fts.spend import SpendCapExceeded, SpendLedger, prices_from_config

REPO_ROOT = Path(__file__).resolve().parent.parent
HOSTED_YAML = REPO_ROOT / "config" / "hosted.yaml"
RESULT_JSON = REPO_ROOT / "results" / "hosted_check.json"
SECTION = "Hosted model and Spend_Cap"
CHECK_MESSAGES = [
    {"role": "system", "content": "You are a connectivity check. Follow the instruction exactly."},
    {"role": "user", "content": "Reply with the single word OK."},
]
CHECK_MAX_TOKENS = 16


def load_config(path: Path | str = HOSTED_YAML) -> dict:
    return settings.load_yaml(path)


def open_ledger(cfg: dict, root: Path = REPO_ROOT) -> SpendLedger:
    return SpendLedger(root / cfg["ledger_path"], cap_usd=float(cfg["spend_cap_usd"]), prices=prices_from_config(cfg))


def open_hosted(cfg: dict, root: Path = REPO_ROOT, tag: str = "", **client_kwargs) -> tuple[ModelClient, SpendLedger]:
    """The ledger and a client that checks and records every request in it."""
    ledger = open_ledger(cfg, root)
    client = client_from_config(cfg, ledger=ledger, **client_kwargs)
    client.ledger_tag = tag
    return client, ledger


def render_section(cfg: dict, check: dict, ledger_summary: dict, stamp: repro.Stamp) -> str:
    prices = cfg["prices_usd_per_1m_tokens"]
    lines = [
        f"- Hosted_Model: `{cfg['model']}` via Chat Completions at `{cfg['base_url']}` with "
        f"`reasoning_effort: {cfg['extra'].get('reasoning_effort')}`; key read from `{cfg['api_key_env']}`",
        f"- API check: the response's `model` field was `{check['response_model']}`; reply `{check['text']}`",
        f"- Parameters sent and accepted: {', '.join(f'`{p}`' for p in check['accepted_params'])}; "
        f"dropped after rejection: {', '.join(f'`{p}`' for p in check['dropped_params']) or 'none'}",
        f"- Check tokens: {check['prompt_tokens']} prompt, {check['completion_tokens']} completion; "
        f"cost ${check['cost_usd']:.6f}",
        f"- Prices (USD per 1M tokens, {cfg['prices_source']}, accessed {cfg['prices_accessed']}): input {prices['input']}, "
        f"cached input {prices['cached_input']}, cache write {prices['cache_write']}, output {prices['output']}; "
        f"the ledger charges prompt tokens at {ledger_summary['prompt_price_per_mtok']} and completion tokens at "
        f"{ledger_summary['completion_price_per_mtok']}",
        f"- **Spend_Cap: ${float(cfg['spend_cap_usd']):.2f}**; ledger `{cfg['ledger_path']}` total after the check: "
        f"${ledger_summary['spend_usd']:.6f} over {ledger_summary['requests']} request(s)",
        *envdoc.stamp_lines(stamp),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, env_md: Path = envdoc.ENVIRONMENT_MD, transport=None) -> int:
    cfg = load_config(root / "config" / "hosted.yaml")
    stamp = repro.start_run(seed=int(cfg.get("seed", 0)))
    try:
        client, ledger = open_hosted(cfg, root, tag="hosted_check", transport=transport)
    except MissingSecretError as exc:
        print(f"FAIL: {exc}; no request was sent", file=sys.stderr)
        return 2
    with client:
        try:
            completion = client.complete(CHECK_MESSAGES, CHECK_MAX_TOKENS)
        except SpendCapExceeded as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        except ModelClientError as exc:
            print(f"FAIL: hosted request failed: {exc}", file=sys.stderr)
            return 1
        check = {
            "configured_model": cfg["model"],
            "response_model": completion.model,
            "text": completion.text.strip(),
            "accepted_params": client.sent_params(CHECK_MAX_TOKENS),
            "dropped_params": list(client.dropped_params),
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "usage_reported": completion.usage_reported,
            "cost_usd": ledger.prices.cost(completion.prompt_tokens, completion.completion_tokens),
        }
    summary = ledger.summary()
    repro.write_json(root / "results" / "hosted_check.json", {"check": check, "ledger": summary}, stamp)
    envdoc.replace_section(SECTION, render_section(cfg, check, summary, stamp), env_md)
    print(f"hosted check OK: model {check['response_model']}, {check['prompt_tokens']}+{check['completion_tokens']} tokens, "
          f"${check['cost_usd']:.6f}; dropped {check['dropped_params'] or 'none'}")
    return 0
