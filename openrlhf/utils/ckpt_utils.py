import os
import shutil
from typing import List

HF_CKPT_SUFFIX = "_hf"
BEST_CKPT_PREFIX = "best"


def rotate_hf_checkpoints(ckpt_path: str, tag: str, max_num: int) -> List[str]:
    """Evict old HF-format checkpoint exports (``{tag}_hf`` dirs under ``ckpt_path``)
    before a new one is saved, mirroring the DeepSpeed-side ``save_ckpt`` rotation:

    - regular exports are kept to at most ``max_num`` (counting the export about to
      be written as ``{tag}_hf``), evicting the oldest by mtime first;
    - ``best*_hf`` exports do not count toward ``max_num``; saving a new best evicts
      the previous best exports instead.

    Must be called from a single rank only. Returns the list of removed directories.
    """
    removed = []
    if not os.path.isdir(ckpt_path):
        return removed

    new_export = f"{tag}{HF_CKPT_SUFFIX}"
    is_best = tag.startswith(BEST_CKPT_PREFIX)

    exports = [
        d
        for d in os.listdir(ckpt_path)
        if d.endswith(HF_CKPT_SUFFIX) and d != new_export and os.path.isdir(os.path.join(ckpt_path, d))
    ]

    if is_best:
        candidates = [d for d in exports if d.startswith(BEST_CKPT_PREFIX)]
    else:
        if max_num is None:
            return removed
        regular = [d for d in exports if not d.startswith(BEST_CKPT_PREFIX)]
        regular.sort(key=lambda d: os.path.getmtime(os.path.join(ckpt_path, d)))
        # +1 accounts for the export about to be saved
        overflow = max(0, len(regular) - max_num + 1)
        candidates = regular[:overflow]

    for d in candidates:
        delete_dir = os.path.join(ckpt_path, d)
        shutil.rmtree(delete_dir, ignore_errors=True)
        removed.append(delete_dir)
    return removed
