"""Pytest bootstrap: ensure the sandbox root (containing the `src` package) is
on sys.path so `import src...` works when pytest is invoked from any cwd.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
