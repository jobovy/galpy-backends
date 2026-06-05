"""CLI entry point: ``python -m paper.figures._harness <run|verify|list>``.

Kept separate from figure.py so the figure registry lives in exactly one module
instance (a figure's make.py imports paper.figures._harness.figure; running
figure.py directly as __main__ would create a second, empty registry).
"""

from .figure import main

if __name__ == "__main__":
    main()
