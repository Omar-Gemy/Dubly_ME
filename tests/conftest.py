"""Shared pytest setup: make src/ importable the same way the CLIs do.

Every phase is executed as ``python src/<phase>.py``, which puts src/ on
sys.path[0] — so the modules import each other flatly (``import pipeline_core``).
Tests reproduce that import environment here instead of in each test file.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
