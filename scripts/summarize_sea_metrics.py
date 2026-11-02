#!/usr/bin/env python3
"""CLI wrapper for final-window SEA metric aggregation."""

import os

# Metric aggregation is NumPy-only. Avoid initializing CUDA through the
# package's eager imports when training processes already occupy the GPU.
os.environ["JAX_PLATFORMS"] = "cpu"

from craftax.sea.summarize_results import main


if __name__ == "__main__":
    main()
