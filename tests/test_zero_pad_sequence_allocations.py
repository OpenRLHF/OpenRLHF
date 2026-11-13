from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from openrlhf.datasets import SFTDataset
from openrlhf.utils.utils import zero_pad_sequences


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("stack", [False, True])
@pytest.mark.parametrize("lengths", [(3,), (0,), (3, 3), (1, 4, 0), (0, 0)])
@pytest.mark.parametrize("dtype", [torch.int64, torch.float64, torch.bool])
def test_padding_values_layouts_and_independent_output(side, stack, lengths, dtype):
    inputs = []
    for i, length in enumerate(lengths):
        storage = torch.arange(4 * length).reshape(2, 2 * length) + i + 1
        inputs.append((storage % 2 if dtype == torch.bool else storage).to(dtype)[:, ::2])
    originals = [tensor.clone() for tensor in inputs]
    max_len = max(lengths)
    pad_value = True if dtype == torch.bool else -2
    expected_items = []
    for original in originals:
        item = torch.full((2, max_len), pad_value, dtype=dtype)
        start = max_len - original.shape[-1] if side == "left" else 0
        item[:, start : start + original.shape[-1]] = original
        expected_items.append(item)
    expected = torch.stack(expected_items) if stack else torch.cat(expected_items)
    actual = zero_pad_sequences(inputs, side=side, value=pad_value, stack=stack)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.fill_(0)
    for original, tensor in zip(originals, inputs):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("stack", [False, True])
@pytest.mark.parametrize("lengths", [(5,), (2, 5)])
def test_zero_width_padding_preserves_autograd(side, stack, lengths):
    inputs = [torch.arange(2 * length, dtype=torch.float64).reshape(2, length).requires_grad_() for length in lengths]
    references = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    padded = []
    for tensor in references:
        missing = 5 - tensor.shape[-1]
        padded.append(F.pad(tensor, (missing, 0) if side == "left" else (0, missing), value=-2))
    expected = torch.stack(padded) if stack else torch.cat(padded)
    actual = zero_pad_sequences(inputs, side=side, value=-2, stack=stack)
    weights = torch.arange(actual.numel(), dtype=torch.float64).reshape(actual.shape)
    actual_gradients = torch.autograd.grad((actual * weights).sum(), inputs)
    expected_gradients = torch.autograd.grad((expected * weights).sum(), references)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("stack", [False, True])
def test_single_sequence_batches_allocate_only_the_combined_output(stack):
    inputs = [torch.arange(8 * 4096, dtype=torch.float32).reshape(8, 4096)]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
        result = zero_pad_sequences(inputs, side="right", stack=stack)
    allocated = sum(max(event.self_cpu_memory_usage, 0) for event in profile.events())
    output_bytes = result.numel() * result.element_size()
    # The final cat/stack still owns its output; a single sequence needs no
    # additional full-sized padding clone before that allocation.
    assert allocated <= output_bytes + 4096, (allocated, output_bytes)
    for tensor in inputs:
        assert result.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()


@pytest.mark.parametrize("stack", [False, True])
@pytest.mark.parametrize("layout", ["transpose", "expanded", "channels_last"])
def test_single_sequence_preserves_output_layout_and_ownership(stack, layout):
    if layout == "transpose":
        tensor = torch.arange(24).reshape(2, 3, 4).transpose(1, 2)
    elif layout == "expanded":
        tensor = torch.arange(6).expand(3, -1)
    else:
        tensor = torch.arange(120).reshape(2, 3, 4, 5).contiguous(memory_format=torch.channels_last)
    original = tensor.clone()
    padded = F.pad(tensor, (0, 0), value=7)
    expected = torch.stack([padded]) if stack else torch.cat([padded])
    actual = zero_pad_sequences([tensor], side="right", value=7, stack=stack)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.stride() == expected.stride()
    actual.fill_(0)
    torch.testing.assert_close(tensor, original, rtol=0, atol=0)


@pytest.mark.parametrize("pretrain_mode", [False, True])
@pytest.mark.parametrize("micro_batch_size", [1, 2])
def test_native_sft_collation_preserves_optimizer_updates(pretrain_mode, micro_batch_size):
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, **{f"t{i}": i + 3 for i in range(16)}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    data = Dataset.from_dict(
        {"text": ["t1 t2 ", "t3 t4 ", "t5 t6 t7 ", "t8 t9 "], "answer": ["t3 t4", "t5 t6", "t8 t9", "t10 t11"]}
    )
    strategy = SimpleNamespace(
        args=SimpleNamespace(data=SimpleNamespace(input_key="text", output_key="answer", apply_chat_template=False))
    )
    dataset = SFTDataset(data, tokenizer, 32, strategy, pretrain_mode=pretrain_mode, num_processors=1)
    loader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    torch.manual_seed(23)
    config = GPT2Config(
        vocab_size=len(vocabulary),
        n_positions=32,
        n_embd=8,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    reference = GPT2LMHeadModel(config)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for step in range(3):
        actual = list(loader)[step % len(loader)]
        first = (step % len(loader)) * micro_batch_size
        items = [dataset[index] for index in range(first, min(first + micro_batch_size, len(dataset)))]
        expected = []
        for column, value in zip(zip(*items), (tokenizer.pad_token_id, 0, 0)):
            max_length = max(tensor.shape[-1] for tensor in column)
            expected.append(
                torch.cat([F.pad(tensor, (0, max_length - tensor.shape[-1]), value=value) for tensor in column])
            )
        for observed, original in zip(actual, expected):
            torch.testing.assert_close(observed, original, rtol=0, atol=0)
        for network, opt, batch in ((model, optimizer, actual), (reference, reference_optimizer, expected)):
            ids, attention, loss_mask = batch
            opt.zero_grad()
            logits = network(ids, attention_mask=attention).logits[:, :-1]
            losses = F.cross_entropy(logits.reshape(-1, len(vocabulary)), ids[:, 1:].reshape(-1), reduction="none")
            weights = loss_mask[:, :-1].reshape(-1)
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            opt.step()
        for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, ref_parameter, rtol=0, atol=0)
    assert any(not torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters())
