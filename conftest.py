"""Root conftest — ensures the project root is on sys.path for imports."""

import sys
from pathlib import Path

# Add project root to sys.path so all packages are importable
sys.path.insert(0, str(Path(__file__).parent))
