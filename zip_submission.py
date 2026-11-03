"""Create the submission only after its required artifacts pass the audit."""
from pathlib import Path
import subprocess
import sys
raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).parent / 'code/package_submission.py'), *sys.argv[1:]]))
