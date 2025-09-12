
# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass, asdict
from typing import List, Dict

# =========================
# Hyper-parameters (tunable)
# =========================
GAMMA = 0.05   # weight on -log(P_hat)   -> E_param = A_hat - GAMMA * ln(P_hat)
DELTA = 0.05   # weight on -log(D_hat)   -> E_data  = A_hat - DELTA * ln(D_hat)
ALPHA = 100.0    # weight on E_param^ETA
BETA  = 100.0    # weight on E_data^KAPPA
ETA   = 1.0    # exponent for E_param
KAPPA = 1.0    # exponent for E_data
P_MIN = 0.10   # lower bound for P_hat to keep ln(P_hat) finite/stable

# =========================
# Utility score definitions
# =========================
@dataclass
class Row:
    method: str
    modality: str
    size_mb: float
    acc: float  # KITTI: mean AP (Moderate); nuScenes: NDS

@dataclass
class ScoredRow(Row):
    A_hat: float
    P_hat: float
    D_hat: float
    term_param: float
    term_data: float
    E_param: float
    E_data: float
    U_a: float

def compute_scores(
    rows: List[Row],
    d_hat_for_modality: Dict[str, float],
    gamma: float = GAMMA,
    delta: float = DELTA,
    alpha: float = ALPHA,
    beta: float = BETA,
    eta: float = ETA,
    kappa: float = KAPPA,
    p_min: float = P_MIN,
):
    """Compute Accuracy Utility for a dataset and return list of ScoredRow."""
    if not rows:
        return []

    A_max = max(r.acc for r in rows)
    P_max = max(r.size_mb for r in rows)

    out: List[ScoredRow] = []
    for r in rows:
        A_hat = r.acc / 100 if A_max > 0 else 0.0
        P_hat_raw = r.size_mb / 100  if P_max > 0 else 1.0 # P_max
        print()
        P_hat = max(P_hat_raw, p_min)  # clamp to avoid log blow-up, make sure it over 100 percenet 
        D_hat = d_hat_for_modality.get(r.modality, 1.0) # for KITTI  

        term_param = gamma * math.log(P_hat)+2 # here is a Bails 
        term_data  = delta * math.log(D_hat)+2

        E_param = A_hat / term_param
        E_data  = A_hat / term_data
        U_a     = (alpha * (E_param ** eta)) + (beta * (E_data ** kappa))

        out.append(
            ScoredRow(
                method=r.method, modality=r.modality, size_mb=r.size_mb, acc=r.acc,
                A_hat=A_hat, P_hat=P_hat, D_hat=D_hat,
                term_param=term_param, term_data=term_data,
                E_param=E_param, E_data=E_data, U_a=U_a
            )
        )
        # print('A_hat:', A_hat)
        # print('P_hat:', P_hat)
        # print('D_hat:', D_hat)
        # print('term_param:', term_param)
        # print('term_data:', term_data)
    # Sort by utility (desc)
    out.sort(key=lambda x: x.U_a, reverse=True)
    return out

def print_table(title: str, scored: List[ScoredRow]):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    header = (
        f"{'Rank':>4}  {'Method':<22} {'Modality':<12} "
        f"{'Size(MB)':>8}  {'Acc':>7}  {'Â':>7}  {'P̂':>7}  {'D̂':>5}  "
        f"{'term_p':>8}  {'term_d':>8}  {'E_param':>8}  {'E_data':>8}  {'U_a':>8}"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(scored, 1):
        print(
            f"{i:>4}  {r.method:<22} {r.modality:<12} "
            f"{r.size_mb:8.1f}  {r.acc:7.2f}  {r.A_hat:7.4f}  {r.P_hat:7.4f}  {r.D_hat:5.2f}  "
            f"{r.term_param:8.4f}  {r.term_data:8.4f}  {r.E_param:8.4f}  {r.E_data:8.4f}  {r.U_a:8.4f}"
        )

# =========================
# Fixed raw tables (from paper figures)
# KITTI uses NEW mean AP (Moderate) you provided.
# nuScenes uses NDS from Table 7.
# =========================

# KITTI (mean AP over Ped/Car/Cyc Moderate) + sizes + modality
kitti_rows = [
    Row("EMOS (Ours)",   "Multimodal",  62, 81.74),
    Row("TED",           "Multimodal",  65, 79.88),
    Row("VFF",           "Multimodal", 229, 78.42),
    Row("LoGoNet",       "Multimodal", 266, 77.02),
    Row("M3DETR",        "LiDAR-only", 166, 76.99),
    Row("PartA2-Free",   "LiDAR-only", 226, 76.68),
    Row("PVD",           "LiDAR-only", 147, 75.59),
    Row("PartA2-Anchor", "LiDAR-only", 244, 73.76),
    Row("CT3D",          "LiDAR-only",  30, 73.13),
    Row("EPNet",         "Multimodal", 179, 73.04),
    Row("Second-IoU",    "LiDAR-only",  46, 72.02),
    Row("Second",        "LiDAR-only",  20, 69.36),
    Row("PointPillar",   "LiDAR-only",  18, 66.69),
]

# nuScenes (use NDS as accuracy) + sizes + modality  (Table 7)
nus_rows = [
    Row("PointPillar-MultiHead",    "LiDAR-only",  23.0, 58.2),
    Row("Second-MultiHead",         "LiDAR-only",  35.0, 62.3),
    Row("CenterPoint-PointPillar",  "LiDAR-only",  23.0, 60.7),
    Row("CenterPoint (voxel=0.1)",  "LiDAR-only",  34.0, 64.5),
    Row("CenterPoint (voxel=0.075)","LiDAR-only",  34.0, 66.5),
    Row("VoxelNeXt",                "LiDAR-only",  31.0, 66.6),
    Row("Sparse-DT4 (Camera)",      "Camera-only", 675.0, 54.4),  # treat as unimodal for D
    Row("RCBevdet",                 "Multimodal",  330.7, 56.8),
    Row("CMT",                      "Multimodal",  992.0, 72.9),
    Row("TransFusion",              "Multimodal",  333.9, 71.7),
    Row("BEVDet (base)",            "Multimodal",  401.0, 73.6),
    Row("BEVDet (large)",           "Multimodal",  496.0, 74.0),
    Row("EMOS (Ours)",              "Multimodal",  333.3, 74.5),
]

# D-hat lookup for each dataset
kitti_D = {"Multimodal": 1.0, "LiDAR-only": 0.71, "Camera-only": 0.71}
nus_D   = {"Multimodal": 1.0, "LiDAR-only": 0.44, "Camera-only": 0.56}

# =========================
# Run & print
# =========================
if __name__ == "__main__":
    kitti_scored = compute_scores(kitti_rows, d_hat_for_modality=kitti_D)
    print_table("KITTI – Accuracy Utility (metric=mean AP (Moderate))", kitti_scored)

    nus_scored = compute_scores(nus_rows, d_hat_for_modality=nus_D)
    print_table("nuScenes – Accuracy Utility (metric=NDS)", nus_scored)

    # 提示：如需快速调参，只改顶部的超参即可。
