
# -*- coding: utf-8 -*-
"""
Efficiency Utility scorer for KITTI & nuScenes.
- Uses the utility: U_e = H * (1 - (P - Pmin)/(Pmax - Pmin))^nu + (t_norm/p_norm)^theta
- t_norm = latency / min_latency    (after estimating Jetson time when missing)
- p_norm = size    / min_size
- Prints tables to stdout, sorted by U_e (desc).

You only need to tweak the H, nu, theta, and the latency estimation multipliers below.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import math

# ============
# Hyperparams (tunable)
# ============
H      = 4     # weight for the size-based term
NU     = 4     # exponent for the size-based term
THETA  = -1.0     # exponent for (t_norm / p_norm)

# Estimation multipliers to turn non-Jetson latency into "Jetson-equivalent"
MULT_3080_TO_JETSON  = 3.25   # if only 3080 time is known
MULT_A4000_TO_JETSON = 4.35   # if only A4000 time is known

# When a model has NO latency anywhere (rare), you can optionally skip or set a default
SKIP_IF_NO_LATENCY = True
DEFAULT_JETSON_MS  = 999999.0  # used only if SKIP_IF_NO_LATENCY=False

# ============
# Data schemas
# ============
@dataclass
class ModelRow:
    method: str
    size_mb: float
    jetson_ms: Optional[float] = None
    ms_3080: Optional[float] = None
    ms_a4000: Optional[float] = None

@dataclass
class ScoredRow(ModelRow):
    jetson_est_ms: float = 0.0
    p_norm: float = 0.0
    t_norm: float = 0.0
    size_gain: float = 0.0    # (1 - minmax(size))^NU
    ratio_term: float = 0.0   # (t_norm / p_norm)^THETA
    U_e: float = 0.0

# ============
# Raw tables (from the two figures)
# ============

# --- KITTI (Table 6) ---
kitti_rows: List[ModelRow] = [
    # LiDAR-only
    ModelRow("M3DETR",              166, jetson_ms=None, ms_3080=1157, ms_a4000=338),
    ModelRow("PV-RCNN",               50, jetson_ms=1787, ms_3080=207,  ms_a4000=45),
    ModelRow("Voxel-RCNN (LiDAR)",    28, jetson_ms=None, ms_3080=240,  ms_a4000=70),
    ModelRow("GLENet-VR",             87, jetson_ms=None, ms_3080=575,  ms_a4000=165),
    ModelRow("PartA2-Anchor",        244, jetson_ms=None, ms_3080=343,  ms_a4000=95),
    ModelRow("PartA2-Free",          226, jetson_ms=1064, ms_3080=335,  ms_a4000=124),
    ModelRow("PointPillar",           18, jetson_ms=972,  ms_3080=135,  ms_a4000=41),
    ModelRow("SECOND",                20, jetson_ms=1322, ms_3080=168,  ms_a4000=45),
    ModelRow("Second-IoU",            46, jetson_ms=None, ms_3080=188,  ms_a4000=58),
    ModelRow("CT3D",                  30, jetson_ms=None, ms_3080=174,  ms_a4000=60),
    ModelRow("PVD",                  147, jetson_ms=None, ms_3080=312,  ms_a4000=100),
    # Multimodal
    ModelRow("Voxel-RCNN (MM)",      129, jetson_ms=965,  ms_3080=1057, ms_a4000=331),
    ModelRow("LoGoNet",              266, jetson_ms=None, ms_3080=342,  ms_a4000=100),
    ModelRow("VFF",                  226, jetson_ms=None, ms_3080=584,  ms_a4000=192),
    ModelRow("EPNet",                179, jetson_ms=None, ms_3080=584,  ms_a4000=154),
    ModelRow("TED",                   65, jetson_ms=None, ms_3080=305,  ms_a4000=100),
    ModelRow("EMOS (Ours)",           62, jetson_ms=372.5,ms_3080=295,  ms_a4000=93),
]

# --- nuScenes (Table 7) ---
nuscenes_rows: List[ModelRow] = [
    # LiDAR-only
    ModelRow("PointPillar-MultiHead",   23, jetson_ms=456),
    ModelRow("Second-MultiHead",        35, jetson_ms=5546),
    ModelRow("CenterPoint-PointPillar", 23, jetson_ms=457),
    ModelRow("CenterPoint (voxel=0.1)", 34, jetson_ms=535),
    ModelRow("CenterPoint (voxel=0.075)",34, jetson_ms=535),
    ModelRow("VoxelNeXt",               31, jetson_ms=None),
    # Camera-only
    ModelRow("Sparse-DT4",             675, jetson_ms=None),
    # Multimodal
    ModelRow("RCBevdet",              330.7, jetson_ms=6094.8),
    ModelRow("CMT",                   992.0, jetson_ms=None),
    ModelRow("TransFusion",           333.9, jetson_ms=6172.4),
    ModelRow("BEVDet (base)",         401.0, jetson_ms=7321.5),
    ModelRow("BEVDet (large)",        496.0, jetson_ms=8306.4),
    ModelRow("EMOS (Ours)",           333.3, jetson_ms=397.3),
]

# ============
# Core scoring
# ============
def estimate_jetson_time(r: ModelRow) -> Optional[float]:
    """Return Jetson-equivalent latency."""
    if r.jetson_ms is not None:
        return r.jetson_ms
    if r.ms_3080 is not None:
        return r.ms_3080 * MULT_3080_TO_JETSON
    if r.ms_a4000 is not None:
        return r.ms_a4000 * MULT_A4000_TO_JETSON
    return None

def score_dataset(name: str, rows: List[ModelRow]) -> List[ScoredRow]:
    # 1) Prepare Jetson-equivalent latency and handle missing rows
    prepared: List[ScoredRow] = []
    for r in rows:
        jt = estimate_jetson_time(r)
        if jt is None:
            if SKIP_IF_NO_LATENCY:
                continue
            jt = DEFAULT_JETSON_MS
        prepared.append(ScoredRow(**r.__dict__, jetson_est_ms=jt))

    if not prepared:
        print(f"[WARN] No rows left for {name} after filtering.")
        return []

    # 2) Compute min/max for normalizations
    sizes  = [x.size_mb for x in prepared]
    # print('sizes:', sizes)
    times  = [x.jetson_est_ms for x in prepared]
    Pmin, Pmax = min(sizes), max(sizes)
    Tmin       = min(times)

    # 3) Compute per-row utilities
    for x in prepared:
        # min-max size term (bigger is better when closer to Pmin)
        if Pmax == Pmin:
            minmax = 0.0
        else:
            minmax = (x.size_mb - Pmin) / (Pmax - Pmin)
        x.size_gain = (1.0 - minmax) ** NU

        # normalized ratios
        x.p_norm = x.size_mb / 20
        x.t_norm = x.jetson_est_ms / Tmin
        x.ratio_term = (x.t_norm / x.p_norm) ** THETA

        # x.U_e = H * x.size_gain + x.ratio_term
        x.U_e =  10 * (-1*math.log(x.p_norm) + x.ratio_term +4)# -x.p_norm 
        # print('x.p_norm:', x.p_norm)
        # print('x.t_norm:', x.t_norm)
        # print('x.size_gain:', x.size_gain)
        # print('x.ratio_term :', x.ratio_term )
        # print('H:', H)
        # print('x.U_e:', x.U_e)

    # 4) Sort by utility (desc)
    prepared.sort(key=lambda z: z.U_e, reverse=True)
    return prepared

def print_table(title: str, scored: List[ScoredRow]):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    hdr = (
        f"{'Rank':>4}  {'Method':<24} {'Size(MB)':>8}  "
        f"{'Jetson* (ms)':>12}  {'p_norm':>8}  {'t_norm':>8}  "
        f"{'size_gain':>9}  {'ratio':>9}  {'U_e':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(scored, 1):
        print(
            f"{i:>4}  {r.method:<24} {r.size_mb:8.1f}  "
            f"{r.jetson_est_ms:12.1f}  {r.p_norm:8.3f}  {r.t_norm:8.3f}  "
            f"{r.size_gain:9.3f}  {r.ratio_term:9.3f}  {r.U_e:9.3f}"
        )

# ============
# Run
# ============
if __name__ == "__main__":
    kitti_scored = score_dataset("KITTI", kitti_rows)
    print_table(
        f"KITTI – Efficiency Utility "
        f"(H={H}, nu={NU}, theta={THETA}; 3080→Jetson×{MULT_3080_TO_JETSON}, A4000→Jetson×{MULT_A4000_TO_JETSON})",
        kitti_scored
    )

    nus_scored = score_dataset("nuScenes", nuscenes_rows)
    print_table(
        f"nuScenes – Efficiency Utility "
        f"(H={H}, nu={NU}, theta={THETA}; 3080→Jetson×{MULT_3080_TO_JETSON}, A4000→Jetson×{MULT_A4000_TO_JETSON})",
        nus_scored
    )
