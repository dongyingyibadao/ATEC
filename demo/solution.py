try:
    from .my_refactored.entry import AlgSolution  # noqa: F401
except ImportError:
    from my_refactored.entry import AlgSolution  # noqa: F401
