"""Sequence generation (sampling) from a trained model.

# TODO(owner: B) — implement generate().
"""

# TODO(owner: B)

from __future__ import annotations


def generate(
    model,
    num_sequences: int = 100,
    max_length: int = 130,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    seed: int = 42,
) -> list[str]:
    """Sample sequences from a trained model.

    Parameters
    ----------
    model:
        A trained ``torch.nn.Module`` returned by :func:`src.model.build_model`.
    num_sequences:
        Number of sequences to generate.
    max_length:
        Maximum sequence length (in tokens, excluding special tokens).
    temperature:
        Sampling temperature (1.0 = no scaling).
    top_k:
        If > 0, keep only the top-k logits.
    top_p:
        Nucleus sampling threshold (0.9 keeps tokens summing to 90% probability).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    list[str]
        List of generated amino-acid strings.

    Raises
    ------
    NotImplementedError
        Until Team B implements this function.
    """
    raise NotImplementedError  # TODO(owner: B)
