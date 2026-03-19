from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class RunResult:
    ok: bool
    error: str | None = None


def run_python_tests(code: str, tests: str) -> RunResult:
    """
    在受限命名空间中执行生成代码与测试用例。
    注意：这是研究/本地使用的最简执行器，不做强沙箱隔离。
    """
    g = {"__name__": "__main__"}
    l = {}
    try:
        exec(code, g, l)
    except Exception as e:  # noqa: BLE001
        return RunResult(ok=False, error=f"code_exec_error: {e!r}")

    ns = SimpleNamespace(**g, **l)
    try:
        exec(tests, {"ns": ns})
    except Exception as e:  # noqa: BLE001
        return RunResult(ok=False, error=f"tests_failed: {e!r}")

    return RunResult(ok=True)

