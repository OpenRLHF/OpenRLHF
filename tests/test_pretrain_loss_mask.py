"""Pretraining targets must stay within each sequence, regardless of batch padding."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "openrlhf"
    for name in ("models", "utils", "datasets"):
        package = types.ModuleType(f"openrlhf.{name}")
        package.__path__ = [str(root / name)]
        monkeypatch.setitem(sys.modules, package.__name__, package)
    loaded = {}
    for name in ("models.utils", "models.loss", "utils.utils", "utils.loss_utils", "datasets.sft_dataset"):
        spec = importlib.util.spec_from_file_location(f"openrlhf.{name}", root / (name.replace(".", "/") + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "a": 2, "b": 3, "c": 4, "d": 5}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]")


@pytest.mark.parametrize(
    "texts,max_length",
    [
        (["a b", "a b c d"], 16),
        (["a b c d", "a b"], 16),
        (["a b", "c d"], 16),
        (["a b c"], 16),
        (["a", "a b c"], 16),
        (["", "a b c"], 16),
        (["a b c d", "a b"], 3),
        (["a [EOS]", "a [EOS] b c"], 16),
    ],
)
@pytest.mark.parametrize("token_level", [True, False])
def test_pretrain_targets_and_gradients_follow_valid_next_tokens(modules, tokenizer, texts, max_length, token_level):
    strategy = SimpleNamespace(args=SimpleNamespace(data=SimpleNamespace(input_key="text", output_key=None)))
    dataset = modules["datasets.sft_dataset"].SFTDataset(
        Dataset.from_dict({"text": texts}), tokenizer, max_length, strategy, pretrain_mode=True, num_processors=1
    )
    items = [dataset[i] for i in range(len(dataset))]
    for item, text in zip(items, texts):
        expected_ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        assert item[0].tolist() == [expected_ids]
    ids, attention, mask = [x.squeeze(1) for x in dataset.collate_fn(items)]
    # A real source position AND a real next token are necessary for a causal target.
    expected = attention[:, :-1].bool() & attention[:, 1:].bool()
    torch.testing.assert_close(mask[:, :-1].bool(), expected)
    counts = expected.sum(-1)
    torch.manual_seed(42)
    logits = torch.randn(*ids.shape, len(tokenizer), requires_grad=True)
    logps = -F.cross_entropy(
        logits[:, :-1].reshape(-1, len(tokenizer)), ids[:, 1:].reshape(-1), reduction="none"
    ).reshape_as(expected)
    actual = modules["models.loss"].SFTLoss(token_level_loss=token_level)(logps, mask[:, :-1])
    if expected.any():
        expected_loss = (
            -logps[expected].mean()
            if token_level
            else torch.stack([-logps[i, :count].mean() for i, count in enumerate(counts) if count]).mean()
        )
        torch.testing.assert_close(actual, expected_loss)
        expected_grad = torch.autograd.grad(expected_loss, logits, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(actual, logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        info = modules["utils.loss_utils"].get_loss_batch_info(strategy, mask[:, :-1])
        assert info["batch_num_tokens"] == expected.sum()
        assert info["global_batch_size"] == (counts > 0).sum()


def test_ordinary_sft_and_multiturn_masks_stay_unchanged(modules):
    dataset = modules["datasets.sft_dataset"].SFTDataset.__new__(modules["datasets.sft_dataset"].SFTDataset)
    dataset.pretrain_mode = False
    dataset.multiturn = False
    dataset.prompt_ids_lens = [2]
    ids = torch.tensor([[2, 3, 4, 5, 1]])
    assert dataset.get_loss_mask(ids, 0).tolist() == [[0, 1, 1, 1, 0]]
    dataset.multiturn = True
    dataset.response_ranges = [[(2, 2), (4, 4)]]
    assert dataset.get_loss_mask(ids, 0).tolist() == [[0, 1, 0, 1, 0]]


def test_sft_mask_includes_boundary_merged_response_token(modules):
    text = "User: Write Python\nAssistant: def solution(): pass"
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator([text], trainers.BpeTrainer(vocab_size=100, special_tokens=["[UNK]", "[EOS]"]))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]"
    )
    strategy = SimpleNamespace(
        args=SimpleNamespace(
            data=SimpleNamespace(input_key="question", output_key="response", apply_chat_template=False)
        )
    )
    dataset = modules["datasets.sft_dataset"].SFTDataset(
        Dataset.from_dict({"question": ["Write Python"], "response": ["def solution(): pass"]}),
        tokenizer,
        64,
        strategy,
        input_template="User: {}\nAssistant: ",
        num_processors=1,
    )
    _, _, mask = dataset[0]
    prompt = dataset.prompts[0]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    response_token = next(i for i, (_, end) in enumerate(encoded["offset_mapping"]) if end > len(prompt))
    assert tokenizer.convert_ids_to_tokens(encoded["input_ids"])[response_token] == "Ġdef"
    assert mask[0, response_token - 1] == 1


@pytest.mark.parametrize("length", [0, 1])
def test_pretrain_sequence_without_next_token_has_no_targets(modules, length):
    dataset = modules["datasets.sft_dataset"].SFTDataset.__new__(modules["datasets.sft_dataset"].SFTDataset)
    dataset.pretrain_mode = True
    mask = dataset.get_loss_mask(torch.zeros((1, length), dtype=torch.long), 0)
    assert mask.shape == (1, length)
    assert mask.sum() == 0
