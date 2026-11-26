"""FlashAttention in a form pip can build unattended.

The release on PyPI cannot be: it imports PyTorch while building without declaring it as a build
requirement, and it compiles with -std=c++17, which the headers of PyTorch 2.13+ reject. This
builds that release unchanged otherwise, against the PyTorch pinned in pyproject.toml, and makes
it require the same PyTorch at run time.
"""

import os
import tarfile
import urllib.request

import psutil
import torch

VERSION = "2.8.3.post2"
# A compile job peaks far above the 9 GB the release budgets for it, and too many at once
# exhaust the memory, so the number of jobs is chosen here unless MAX_JOBS already says.
if "MAX_JOBS" not in os.environ:
    os.environ["MAX_JOBS"] = str(max(1, min(os.cpu_count() // 2, psutil.virtual_memory().available // (24 * 2**30))))
archive, _ = urllib.request.urlretrieve(
    f"https://files.pythonhosted.org/packages/source/f/flash_attn/flash_attn-{VERSION}.tar.gz"
)
with tarfile.open(archive) as tar:
    tar.extractall(filter="data")
os.chdir(f"flash_attn-{VERSION}")
with open("setup.py") as f:
    source = f.read()
source = source.replace("-std=c++17", "-std=c++20")
source = source.replace('"torch",', f'"torch=={torch.__version__.split("+")[0]}",')
path = os.path.abspath("setup.py")
exec(compile(source, path, "exec"), {"__file__": path, "__name__": "__main__"})
