"""Keep the documented CLI examples in sync with the argparse definitions.

Users copy the README quickstart commands verbatim, so a flag the CLI no longer
accepts makes the documented command fail at startup with "unrecognized
arguments". The parsers cannot be imported on CPU (they pull in ray/vllm), so
the accepted flags are read from the CLI sources with ``ast`` instead.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI_DIR = ROOT / "openrlhf" / "cli"
DOCS = ["README.md", "README_zh.md", "README_ja.md"]

CLI_MARKER = re.compile(r"openrlhf\.cli\.(\w+)")
FLAG = re.compile(r"(?<![-\w])(--[A-Za-z][A-Za-z0-9_.\-]*)")
BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def _prefixes(tree):
    """Return the values a `prefix` loop variable iterates over."""
    values = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "prefix":
            if isinstance(node.iter, (ast.Tuple, ast.List)):
                values.update(e.value for e in node.iter.elts if isinstance(e, ast.Constant))
    return values


def _accepted_flags(path):
    """Collect every `--flag` the argparse parser in `path` accepts."""
    tree = ast.parse(path.read_text())
    prefixes = _prefixes(tree)
    flags = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            flags.add(first.value)
        elif isinstance(first, ast.JoinedStr):
            template = "".join(v.value if isinstance(v, ast.Constant) else "{prefix}" for v in first.values)
            flags.update(template.replace("{prefix}", p) for p in prefixes)
    return flags


def _flags_in(text):
    """Return the `--flags` mentioned in `text`."""
    return {m.group(1) for m in FLAG.finditer(text)}


def _documented_commands(text):
    """Yield (module, flags) for every documented `openrlhf.cli.<module>` command.

    Only the flags that follow the module marker belong to the CLI; everything
    before it (e.g. `deepspeed --module`, `ray job submit --address`) is a
    launcher option.
    """
    for block in BASH_BLOCK.findall(text):
        lines = block.splitlines()
        for index, line in enumerate(lines):
            marker = CLI_MARKER.search(line)
            if not marker:
                continue
            module = marker.group(1)
            segments = [line[marker.end() :]]
            end = index
            while segments[-1].rstrip().endswith("\\"):
                end += 1
                if end >= len(lines):
                    break
                segments.append(lines[end])
            flags = set()
            for segment in segments:
                flags |= _flags_in(segment)
            # Option lists commented out below the command document the same CLI.
            for comment in lines[end + 1 :]:
                if not comment.strip().startswith("#"):
                    break
                flags |= _flags_in(comment)
            yield module, flags


@pytest.mark.parametrize("doc", DOCS)
def test_documented_cli_examples_use_accepted_flags(doc):
    path = ROOT / doc
    if not path.exists():
        pytest.skip(f"{doc} is not part of this checkout")

    unknown = []
    for module, flags in _documented_commands(path.read_text()):
        accepted = _accepted_flags(CLI_DIR / f"{module}.py")
        assert accepted, f"no argparse options found for openrlhf.cli.{module}"
        unknown.extend(f"{doc}: --{flag[2:]} is not accepted by openrlhf.cli.{module}" for flag in flags - accepted)

    assert not unknown, "documented CLI examples use flags the CLI does not accept:\n" + "\n".join(unknown)
