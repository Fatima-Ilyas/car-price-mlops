# sync_versions.py - writes requirements.txt / requirements-dev.txt with the EXACT versions
# installed on YOUR laptop, and makes Dockerfile + CI use the SAME Python version.
# Run it after training works:  python sync_versions.py

import re
import sys
from importlib.metadata import version

api_libs = ["scikit-learn", "pandas", "numpy", "joblib", "fastapi", "uvicorn", "prometheus-client"]
with open("requirements.txt", "w") as f:
    for lib in api_libs:
        f.write(f"{lib}=={version(lib)}\n")

with open("requirements-dev.txt", "w") as f:
    f.write("-r requirements.txt\n")
    f.write(f"mlflow=={version('mlflow')}\n")
    f.write(f"dvc[s3]=={version('dvc')}\n")
    f.write(f"pytest=={version('pytest')}\n")
    f.write(f"httpx=={version('httpx')}\n")

py = f"{sys.version_info.major}.{sys.version_info.minor}"

def patch(path, pattern, replacement):
    text = open(path).read()
    open(path, "w").write(re.sub(pattern, replacement, text))

patch("Dockerfile", r"FROM python:[\d.]+-slim", f"FROM python:{py}-slim")
patch(".github/workflows/ml-pipeline.yml", r'python-version: "[\d.]+"', f'python-version: "{py}"')
patch(".github/workflows/ml-pipeline.yml", r"Set up Python [\d.]+", f"Set up Python {py}")

print("requirements.txt and requirements-dev.txt written")
print(f"Dockerfile and ml-pipeline.yml now use Python {py}")
