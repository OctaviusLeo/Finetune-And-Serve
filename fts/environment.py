"""Pure environment helpers for the VRAM dry-run (Req 1.1-1.5, 0.4).

These functions are deliberately free of any GPU or heavy-ML import so they can
be unit-tested on CPU. The GPU measurement itself lives in
``scripts/dry_run.py`` and lazy-imports the heavy stack inside its function.
"""

from __future__ import annotations

import re

#: Minimum supported Python version (major, minor). Req: Python 3.11+.
MIN_PYTHON: tuple[int, int] = (3, 11)


def python_version_ok(version_str: str, minimum: tuple[int, int] = MIN_PYTHON) -> bool:
    """Return True when ``version_str`` parses to a version >= ``minimum``.

    ``version_str`` is a dotted version like ``"3.11.5"`` or ``"3.11"``. Only
    the major and minor components are compared. Malformed input returns False.
    """
    match = re.match(r"^\s*(\d+)\.(\d+)", version_str)
    if match is None:
        return False
    major, minor = int(match.group(1)), int(match.group(2))
    return (major, minor) >= minimum


def vram_within_budget(peak_mib: float, total_mib: float) -> bool:
    """Return True when measured ``peak_mib`` does not exceed ``total_mib``.

    A non-positive ``total_mib`` means no budget is available, so nothing fits.
    """
    if total_mib <= 0:
        return False
    return peak_mib <= total_mib


def parse_nvidia_smi_total_mib(text: str) -> int | None:
    """Parse the total GPU memory in MiB from ``nvidia-smi`` output.

    Accepts either the CSV form produced by
    ``nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits``
    (a bare integer, optionally with a ``MiB`` suffix) or the human-readable
    table form that contains ``<used>MiB / <total>MiB``. Returns the first total
    found, or ``None`` when no total can be parsed.
    """
    # Human-readable table: "1234MiB / 10240MiB" -> take the total (after '/').
    table = re.search(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", text)
    if table is not None:
        return int(table.group(2))

    # CSV / query form: a line that is just an integer (optionally "MiB").
    for line in text.splitlines():
        stripped = line.strip()
        csv = re.fullmatch(r"(\d+)\s*(?:MiB)?", stripped)
        if csv is not None:
            return int(csv.group(1))

    return None
