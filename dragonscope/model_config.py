"""The small BDH used for the demo; it must run on a CPU-only laptop."""

from dragonscope.vendor.bdh.bdh import BDHConfig


def demo_config(dropout: float = 0.1) -> BDHConfig:
    return BDHConfig(
        n_layer=4,
        n_embd=128,
        n_head=4,
        mlp_internal_dim_multiplier=32,
        dropout=dropout,
        vocab_size=256,
    )
