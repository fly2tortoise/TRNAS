#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Title: A Training-free Robust NAS method for DARTS space
Author: YYM
Description:  zero-cost proxy RScore (SNAP/ConJS) on RobustBench architectures.
"""

import os
import sys
import json
import warnings
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

# Custom modules
from data_c10 import get_c10
from src.utils.utilities import network_weight_gaussian_init
from src.metrics.lap import SNAP
from src.search_space.networks import Network_c, Network_l
from zero_cost_methods.pruners import predictive_js
from darts_space.genotypes import Genotype

# ===========================
# Utility
# ===========================
def exp_normalize(arr, lower=1, upper=100):
    log_data = np.log(arr)  # 
    min_val = np.min(log_data)
    max_val = np.max(log_data)
    scaled = lower + (upper - lower) * (log_data - min_val) / (max_val - min_val)
    return scaled

def load_robustbench_data(base_dir: Path):
    """Load all RobustBench seed JSONs."""
    files = [f"robust_bench_seed_{i}_{i+1000}.json" for i in range(0, 4000, 1000)]
    all_data = []
    for f in files:
        path = base_dir / f
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        with open(path, "r") as fp:
            all_data += json.load(fp)
    print(f"Loaded RobustBench architectures.)")
    return all_data

# ===========================
# Core evaluation
# ===========================
def rscore(args):
    """Compute correlation on RobustBench."""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ---- Dataset ----
    train_data, _, _ = get_c10("cifar10", args.data_path, (args.input_samples, 3, 32, 32), -1)
    train_loader = data.DataLoader(train_data, batch_size=args.input_samples, num_workers=args.num_data_workers, pin_memory=True)
    inputs, _ = next(iter(train_loader))

    # ---- RobustBench ----
    root_dir = Path(__file__).resolve().parents[2] / "0_Data"
    robbench = load_robustbench_data(root_dir)
    robbench = robbench[:args.end]  # Limit the number of architectures for faster testing

    # ---- Storage ----
    proxy_conjs, proxy_snap = [], []
    acc_clean, acc_adv = [], []

    is_tty = sys.stdout.isatty()
    for net_info in tqdm(robbench, disable=not is_tty):

        genotype_info = net_info["net"]["genotype"]
        genotype = Genotype(normal=genotype_info["normal"], normal_concat=genotype_info["normal_concat"], reduce=genotype_info["reduce"], reduce_concat=genotype_info["reduce_concate"],)
        acc_clean.append(net_info["max_test_top1_original"])
        acc_adv.append(net_info["max_test_top1_adv"])

        # ---- Build model ----
        # Avoid potential hidden state contamination between models
        net   = Network_c(3, 10, args.layer_c, genotype).to(device)
        net_l = Network_l(3, 10, args.layer_c, genotype).to(device)

        # Consistency proxy (ConJS)
        # To prevent deep networks from collapsing to zero consistency,
        # remove gradient-based consistency (replaced by activation-based similarity).
        # JS divergence is used to measure parameter and feature consistency.
        proxy_c = predictive_js.trnas_c_no(net, train_loader)

        # Activation proxy (SNAP)
        # To stabilize activation pattern statistics,
        # Gaussian noise is added during feature extraction.
        # This enhances local smoothness and mitigates activation saturation.
        proxy_s = SNAP(model=net_l, inputs=inputs, device=device, seed=args.seed, sap_std=args.sap_std)
        scores = []
        for _ in range(args.repeats):
            net_l.apply(network_weight_gaussian_init)
            proxy_s.reinit()
            scores.append(proxy_s.forward())
            proxy_s.clear()
        proxy_snap.append(np.mean(scores))
        proxy_conjs.append(proxy_c)

    # ---- Correlation evaluation ----
    def corr_report(name, x, y):
        """Compute and print Spearman / Kendall correlation results."""
        sp, kt = stats.spearmanr(x, y)[0], stats.kendalltau(x, y)[0]
        print(f"{name:<10s} SP: {sp:>7.4f},  KT: {kt:>7.4f}")

    # ---- R-Score ----
    snap_norm = exp_normalize(proxy_snap)
    conjs_norm = exp_normalize(proxy_conjs)
    rscore = []
    for i in range(len(snap_norm)):
        fitness_ = np.power(snap_norm[i], 1/2) * conjs_norm[i]
        rscore.append(fitness_)
    corr_report("Rscore", acc_adv, rscore)

# ===========================
# Entry
# ===========================

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser("R-Score: A Training-free Robust NAS method")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--end", type=int, default=223, help="End architecture index.")
    parser.add_argument("--data_path", type=str, default="../../0_Data/", help="Dataset path.")
    parser.add_argument("--repeats", type=int, default=32, help="Repeat times for proxy computation.")
    parser.add_argument("--input_samples", type=int, default=16, help="Input batch size for proxy.")
    parser.add_argument("--layer_l", type=int, default=8, help="Architectural layers for Activation (SNAP).")
    parser.add_argument("--layer_c", type=int, default=8, help="Architectural layers for Consistency (ConJS).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for proxy computation.")
    parser.add_argument("--num_data_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--proxy_types", type=str, default="conjs", help="Proxy type.")
    parser.add_argument("--sap_std", type=float, default=0.05, help="SAP noise std.")
    args = parser.parse_args()

    rscore(args)
