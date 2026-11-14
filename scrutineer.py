#!/usr/bin/env python3
"""
scrutineer.py — Convenience root runner for the Fraud Relay Auditor.
Delegates directly to the authoritative implementation in
.agents/skills/fraud-relay-auditor/scripts/scrutineer.py.
"""
import importlib.util
from pathlib import Path
import sys

_script = (
    Path(__file__).resolve().parent
    / ".agents"
    / "skills"
    / "fraud-relay-auditor"
    / "scripts"
    / "scrutineer.py"
)

if not _script.exists():
    raise FileNotFoundError(f"Auditor script not found at {_script}")

_spec = importlib.util.spec_from_file_location("relay_scrutineer", _script)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["relay_scrutineer"] = _mod
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main()
