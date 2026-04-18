"""Hierarchical argparse helper.

``hierarchize(args)`` converts a flat argparse Namespace whose dest names contain
dots (e.g. ``"muon.lr"``) into a nested SimpleNamespace so callers can write
``args.muon.lr`` instead of ``getattr(args, "muon.lr")``.  Keys without dots stay
at the top level.
"""

from types import SimpleNamespace


def hierarchize(args):
    root = {}
    for k, v in vars(args).items():
        parts = k.split(".")
        node = root
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v

    def build(x):
        if isinstance(x, dict):
            return SimpleNamespace(**{k: build(v) for k, v in x.items()})
        return x

    return build(root)
