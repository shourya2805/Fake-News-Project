import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/shourya2805/Fake-News-Project.git"
WORKDIR = Path("/content/Fake-News-Project")
DATA_URL = "https://raw.githubusercontent.com/DennisIW/FMNV/main/data.json"
AUTHOR_NAME = "Shourya Korrapati"
AUTHOR_EMAIL = "177808621+shourya2805@users.noreply.github.com"
TASKS = {
    "compare": {
        "script": ["compare_encoders.py"],
        "results": ["results/encoder_comparison.json", "results/encoder_comparison.md"],
        "message": "Add frozen text-encoder comparison: 4 encoders x 5 seeds on Colab",
    },
    "confirm": {
        "script": ["confirm_encoder.py"],
        "results": ["results/confirmatory_deberta_vs_bert.json", "results/confirmatory_deberta_vs_bert.md"],
        "message": "Add pre-registered DeBERTa-v3 vs BERT confirmatory test, seeds 47-56 on Colab",
    },
}
CREDENTIAL_HELPER = '!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f'


def run(cmd, env=None, cwd=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def main():
    from google.colab import userdata
    import torch

    task_name = globals().get("TASK", "compare")
    if task_name not in TASKS:
        sys.exit(f"TASK must be one of {sorted(TASKS)}, got {task_name!r}")
    task = TASKS[task_name]
    print(f"task: {task_name}", flush=True)

    if not torch.cuda.is_available():
        sys.exit("no GPU: Runtime > Change runtime type > GPU, then rerun this cell")
    token = userdata.get("GH_TOKEN")
    if not token:
        sys.exit("GH_TOKEN is not set in Colab Secrets (or notebook access is off)")
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", GIT_TERMINAL_PROMPT="0",
               PYTHONUNBUFFERED="1")
    git_env = dict(env, GH_TOKEN=token)

    if (WORKDIR / ".git").exists():
        run(["git", "-C", str(WORKDIR), "pull", "--ff-only"], env=env)
    else:
        run(["git", "clone", REPO, str(WORKDIR)], env=env)
    os.chdir(WORKDIR)
    run(["git", "config", "user.name", AUTHOR_NAME])
    run(["git", "config", "user.email", AUTHOR_EMAIL])
    run(["git", "log", "--oneline", "-1"])

    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-text.txt"], env=env)

    Path("data").mkdir(exist_ok=True)
    run(["curl", "-fsSL", DATA_URL, "-o", "data/data.json"], env=env)
    records = json.loads(Path("data/data.json").read_text())
    print(f"data.json: {len(records)} records, sha256 "
          f"{hashlib.sha256(Path('data/data.json').read_bytes()).hexdigest()}", flush=True)
    if len(records) != 2393:
        sys.exit(f"expected 2393 records, got {len(records)}")
    if subprocess.run(["git", "diff", "--quiet", "--", "data/data.json"]).returncode != 0:
        sys.exit("downloaded data.json differs from the copy tracked in the repo")

    run([sys.executable, *task["script"]], env=env)

    run(["git", "add", *task["results"]])
    run(["git", "commit", "-m", task["message"]])
    run(["git", "-c", "credential.helper=", "-c", f"credential.helper={CREDENTIAL_HELPER}",
         "push", "origin", "HEAD:main"], env=git_env)
    print(Path(task["results"][1]).read_text())


main()
