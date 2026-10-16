"""Model definitions — LSTM and causal Transformer for VHH generation.

# TODO(owner: B) — implement LSTMModel and TransformerModel.
"""

# TODO(owner: B)

from __future__ import annotations

from pathlib import Path


def build_model(config: dict):
    """Build and return the model specified by ``config["model"]["type"]``.

    Parameters
    ----------
    config:
        Parsed YAML config dictionary (see ``configs/vhh_small.yaml``).

    Returns
    -------
    torch.nn.Module
        The constructed model (not yet trained).

    Raises
    ------
    NotImplementedError
        Until Team B implements this function.
    """
    raise NotImplementedError  # TODO(owner: B)


def load_checkpoint(model, ckpt_path: Path):
    """Load model weights from a checkpoint file.

    Raises
    ------
    NotImplementedError
        Until Team B implements this function.
    """
    raise NotImplementedError  # TODO(owner: B)
