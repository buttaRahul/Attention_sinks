import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load(dtype):
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation="sdpa")
    return model.to(DEVICE).eval()


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="session")
def model_bf16():
    return _load(torch.bfloat16)


@pytest.fixture(scope="session")
def model_fp32():
    return _load(torch.float32)


@pytest.fixture(scope="session")
def text_ids(tokenizer):
    """~600 tokens of ordinary English prose, deterministic."""
    para = (
        "The lighthouse keeper climbed the spiral stairs every evening at dusk, counting the steps as his "
        "father had taught him. There were one hundred and twelve of them, worn smooth in the middle by "
        "generations of boots. At the top he trimmed the wick, polished the great lens, and watched the "
        "fishing boats come home across the grey water. Some nights the fog rolled in so thick that he "
        "could not see the rocks below, and on those nights he sounded the horn until morning. "
    )
    ids = tokenizer(para * 8, return_tensors="pt").input_ids
    return ids.to(DEVICE)
