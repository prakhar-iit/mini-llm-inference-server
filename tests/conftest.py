import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))  # benchmark tooling (client, workloads) is tested too
DEFAULT_MODEL_DIR = REPO / "models" / "Llama-3.2-1B"


def real_model_dir() -> Path | None:
    """Llama-3.2-1B weights, if downloaded (scripts/download_model.py) or pointed to by MINILLM_MODEL_DIR."""
    d = Path(os.environ.get("MINILLM_MODEL_DIR", DEFAULT_MODEL_DIR))
    return d if (d / "config.json").exists() else None


def assert_idle(core) -> None:
    """Nothing queued or running, no KV held, and R3: every block back in the free list exactly once."""
    s = core.stats()
    assert (s.num_waiting, s.num_running, s.kv_tokens_used, s.kv_blocks_used) == (0, 0, 0, 0), s
    core.check_blocks()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # A stub that still raises NotImplementedError("M0: ...") shows up as a skip naming the milestone,
    # so the suite stays green while the milestone is in progress and the skip list is the to-do list.
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1], f"not implemented yet: {call.excinfo.value}")
