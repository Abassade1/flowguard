"""Shared test setup: point the app at a throw-away database, never the real one."""

import os
import tempfile
from pathlib import Path

os.environ["FLOWGUARD_DB"] = str(Path(tempfile.mkdtemp()) / "test.db")
