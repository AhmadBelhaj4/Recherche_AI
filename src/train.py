"""Training loop for VHH sequence generation models.

Usage (once implemented):
    python src/train.py --config configs/vhh_small.yaml [--resume]

# TODO(owner: B) — implement the training loop.
"""

# TODO(owner: B)

from __future__ import annotations

import argparse


def train(config: dict, resume: bool = False) -> None:
    """Run one full training run.

    Parameters
    ----------
    config:
        Parsed YAML config dictionary (see ``configs/vhh_small.yaml``).
    resume:
        If ``True``, resume from the last checkpoint at ``CKPT/last.pt``.

    Raises
    ------
    NotImplementedError
        Until Team B implements this function.
    """
    raise NotImplementedError  # TODO(owner: B)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VHH generation model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint."
    )
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume=args.resume)
