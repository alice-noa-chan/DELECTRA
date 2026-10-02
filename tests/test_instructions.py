import torch

from deletcra.instructions import encode_conversation
from deletcra.losses import response_lm_loss


class Tokenizer:
    bos_token_id = 1
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) + 3 for c in text]}


def test_response_mask_alignment_and_eos():
    encoded, reason = encode_conversation(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        Tokenizer(),
        context=40,
    )
    assert reason is None
    ids, labels, visible, count = encoded
    expected = [ord(c) + 3 for c in "Hello"] + [2]
    assert [label for label in labels if label != -100] == expected
    assert ids[visible - 1] == labels[visible - 1] == 2
    assert count == 6
    assert ids[visible:] == [0] * (40 - visible)
    assert labels[visible:] == [-100] * (40 - visible)
    assert labels[0] == -100


def test_instruction_rejects_long_and_unfinished_conversations():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "A" * 50},
    ]
    assert encode_conversation(messages, Tokenizer(), context=30)[1] == "over_context"
    assert encode_conversation(messages[:1], Tokenizer())[1] == "unfinished"
    messages[-1]["content"] = "<think>secret reasoning</think>answer"
    assert encode_conversation(messages, Tokenizer())[1] == "reasoning_or_empty"
    assert encode_conversation([None], Tokenizer())[1] == "schema"
    assert encode_conversation([{}], Tokenizer())[1] == "unfinished"


def test_response_loss_matches_shifted_cross_entropy_and_ignores_prompts():
    torch.manual_seed(7)
    head = torch.nn.Linear(4, 12)
    features = torch.randn(1, 6, 4, requires_grad=True)
    labels = torch.tensor([[-100, -100, -100, 5, 2, -100]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    expected = torch.nn.functional.cross_entropy(
        head(features[:, 2:4]).reshape(2, 12), torch.tensor([5, 2])
    )
    actual = response_lm_loss(head, features, labels, mask)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert features.grad[:, :2].eq(0).all()
    assert features.grad[:, 4:].eq(0).all()
