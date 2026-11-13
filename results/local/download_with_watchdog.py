"""Download HF models, killing and retrying any attempt that stops making progress."""
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
CACHE = Path.home() / ".cache/huggingface/hub"
STALL_SECONDS = 90
# hf_xet hangs on the OpenScholar weights while plain HTTP from the same CDN works,
# so downloads run with HF_HUB_DISABLE_XET=1 (set by the caller).
JOBS = [
    ("OpenSciLM/OpenScholar_Retriever", ["*.json", "*.txt", "pytorch_model.bin"]),
]
PREVIOUS_OK = ["anlp/SciERC_trained_Scibert", "roccas98/Scibert_finetuned_scierc", "anlp/SciERC_trained_SciDeBERTa"]


def model_bytes(repo):
    folder = CACHE / ("models--" + repo.replace("/", "--"))
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) if folder.exists() else 0


for repo in PREVIOUS_OK:
    print("OK", repo, "(already downloaded)", flush=True)
for repo, patterns in JOBS:
    for attempt in range(1, 9):
        for stale in (CACHE / ("models--" + repo.replace("/", "--"))).rglob("*.incomplete"):
            stale.unlink(missing_ok=True)
        code = f"from huggingface_hub import snapshot_download; snapshot_download({repo!r}, allow_patterns={patterns!r})"
        process = subprocess.Popen([PY, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        last_size, last_change = model_bytes(repo), time.time()
        while process.poll() is None:
            time.sleep(5)
            size = model_bytes(repo)
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > STALL_SECONDS:
                process.kill()
                print(f"retry {repo} attempt {attempt}: stalled at {size // 1048576} MB", flush=True)
                break
        if process.returncode == 0:
            print("OK", repo, f"{model_bytes(repo) // 1048576} MB", flush=True)
            break
        if process.returncode not in (None, -9, 1) and process.poll() is not None:
            print(f"retry {repo} attempt {attempt}: exit {process.returncode}", flush=True)
        time.sleep(5)
    else:
        print("FAILED", repo, flush=True)
print("DONE", flush=True)
