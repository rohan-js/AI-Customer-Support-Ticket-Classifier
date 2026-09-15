"""
Pytest configuration: ensure the project root is importable so `utils.*`,
`api.*` and the data generator modules can be imported from tests.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
