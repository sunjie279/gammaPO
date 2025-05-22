# $\gamma$-PO: Robust Preference Optimization via Dynamic Target Margins

[![ACL 2025](https://img.shields.io/badge/ACL-2025-success)](https://aclanthology.org/) [![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/sunjie279/gammaPO)

This repository provides the official implementation of $\gamma$-PO, a novel dynamic target margin preference optimization algorithm for aligning large language models (LLMs) with human preferences. $\gamma$-PO adaptively adjusts target margins at the pairwise level to prioritize high-confidence preference pairs while suppressing noise from ambiguous pairs.

## Key Features

- **Dynamic Margin Adjustment**: Instance-specific margin calibration to handle ambiguous preference pairs.
- **Plug-and-Play Design**: Compatible with variants of Direct Preference Optimization (DPO) that rely on reward margins.
- **State-of-the-Art Performance**: Achieves an average 4.4% improvement over baselines on AlpacaEval2 and Arena-Hard benchmarks.
- **Efficiency**: Minimal code changes required with negligible impact on training efficiency.

## Table of Contents

- [$\\gamma$-PO: Robust Preference Optimization via Dynamic Target Margins](#gamma-po-robust-preference-optimization-via-dynamic-target-margins)
  - [Key Features](#key-features)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Quick Start](#quick-start)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Citation](#citation)

## Introduction

$\gamma$-PO addresses the limitations of existing preference optimization methods by introducing adaptive target margins. This approach strategically prioritizes high-confidence pairs (those demonstrating higher reward margins) while reducing the influence of ambiguous pairs. Through dynamic margin scaling, $\gamma$-PO enhances the alignment of LLMs with human preferences, particularly in the presence of noisy data.

## Quick Start
Our codebase is built upon the [alignment-handbook repo](https://github.com/huggingface/alignment-handbook). The following steps will guide you through the installation process.

First, create a Python virtual environment using e.g. Conda:
```shell
conda create -n handbook python=3.10 && conda activate handbook
```

Next, install PyTorch `v2.2.2`. Since this is hardware-dependent, we
direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/locally/).

You can then install the remaining package dependencies of [alignment-handbook](https://github.com/huggingface/alignment-handbook) as follows:

```shell
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn --no-build-isolation
```

Clone the repository and install dependencies:

```bash
git clone https://github.com/sunjie279/gammaPO.git
cd gammaPO
pip install -r requirements.txt
```

## Data Preparation
Download the UltraFeedback Binarized dataset and prepare preference pairs:
```bash
# Download dataset
huggingface-cli download H4/ultrafeedback_binarized --local-dir data/
```

## Training
Train $\gamma$-PO using the prepared dataset:
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml gammaPO/training_configs/llama-3-8b-it-gmsimpo-beta10-gm0.4-tau10-lr1e-6.yaml
```

## Citation

If you find this repository useful, please cite our ACL 2025 paper:

```bibtex
@inproceedings{sun2025gammaPO,
    title = {$\gamma$-PO: Robust Preference Optimization via Dynamic Target Margins},
    author = {Sun, Jie and Wu, Junkang and Wu, Jiancan and Zhu, Zhibo and Lu, Xingyu and Zhou, Jun and Ma, Lintao and Wang, Xiang},
    booktitle = {Findings of the 63rd Annual Meeting of the Association for Computational Linguistics},
    year = {2025},
    address = {Vienna, Austria}
}
```