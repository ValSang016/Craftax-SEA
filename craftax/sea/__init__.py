"""SEA components for Craftax-Classic.

The original Craftax environments are intentionally left unchanged.  Import the
SEA variants from this package when reproducing the environment used by SEA.
"""

from craftax.sea.env import (
    SeaCraftaxClassicEnvNoAutoReset,
    SeaEnvState,
    make_sea_craftax_classic_env,
)

__all__ = [
    "SeaCraftaxClassicEnvNoAutoReset",
    "SeaEnvState",
    "make_sea_craftax_classic_env",
]
