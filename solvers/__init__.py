"""Five selected solve paths for the trajectory optimization examples."""
__all__ = ["ChainSolver", "cp", "require_icore"]


def __getattr__(name):
    if name == "ChainSolver":
        from .engine import ChainSolver
        return ChainSolver
    raise AttributeError(name)


def require_icore():
    """Fail early with the active interpreter when the solver is not installed."""
    import importlib
    import shlex
    import sys
    from pathlib import Path
    try:
        runtime = importlib.import_module("icore")
        if not all(hasattr(runtime, name) for name in ("Solver", "ChainQP")):
            raise ImportError("The imported icore module has no Solver/ChainQP API")
        return runtime
    except ImportError as error:
        installer = Path(__file__).resolve().parents[1] / "tools/install_icore.py"
        command = shlex.join([sys.executable, str(installer)])
        raise ImportError(
            f"ICORE is unavailable in this Python environment: {sys.executable}\n"
            "Select the 'CT-STL ICORE (Python 3.12)' notebook kernel, or install "
            f"the pinned runtime in this environment:\n{command}\n"
            "Restart the notebook kernel after installation."
        ) from error


try:
    import cvxpy as cp
except ImportError:
    class _MissingCVXPY:
        def __getattr__(self, name):
            raise ImportError(
                "The cvxpy_qoco backend requires CVXPY and QOCO. "
                "Install requirements-comparators.txt or select icore_cpp/icore_c."
            )
    cp = _MissingCVXPY()
