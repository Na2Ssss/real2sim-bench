"""Sim-Ready 3D Asset Post-Processing Pipeline (Mesh Normalization, Hollowing, CoACD, Table Texture)."""

from r2s_bench.assets.sim_ready_pipeline import (
    SimReadyAsset,
    build_sim_ready_asset,
    extract_real_table_texture,
)

__all__ = [
    "SimReadyAsset",
    "build_sim_ready_asset",
    "extract_real_table_texture",
]
