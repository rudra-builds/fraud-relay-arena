#!/usr/bin/env python3
"""
baton_contract.py — Convenience root import for RelayBaton handoff contract.
Delegates directly to the authoritative implementation in
.agents/skills/fraud-relay-auditor/scripts/baton_contract.py.
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
    / "baton_contract.py"
)

if not _script.exists():
    raise FileNotFoundError(f"Baton contract script not found at {_script}")

_spec = importlib.util.spec_from_file_location("baton_contract_core", _script)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["baton_contract_core"] = _mod
_spec.loader.exec_module(_mod)

RelayBaton = _mod.RelayBaton
load_chronological_split = _mod.load_chronological_split
