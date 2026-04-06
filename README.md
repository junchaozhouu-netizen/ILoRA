Artifact Package
================

Overview
--------

This package contains code, configuration files, and dependency specifications for reproducing experiments on federated learning with low-rank adaptation under heterogeneous client settings.

Contents
--------

- computer vision experiments
- natural language processing experiments
- ablation studies
- helper scripts for batch execution

Environment Setup
-----------------

Use Python 3.9 and a CUDA-compatible PyTorch environment when GPU execution is required.

Example setup:

```bash
conda create --name artifact_env python=3.9.6
conda activate artifact_env
pip install -r requirements.txt
```

Running Experiments
-------------------

Launch scripts are organized by task type. Adjust arguments according to the target dataset, client count, optimizer, and heterogeneity setting.

Example:

```bash
python scripts/nlp/run_nlp_adamw_0_5.py \
    --dataset AG_NEWS \
    --num_clients 6 \
    --alpha 0.5 \
    --heterogeneous_rank_clients 2,4,4,6,6,8 \
    --save_dir ./runs/nlp/example_run
```

Logging
-------

Default output directory:

```bash
./runs/
```

TensorBoard:

```bash
python -m tensorboard.main --logdir ./runs
```

Notes
-----

- Prepare datasets in the locations expected by the selected script.
- Save logs and checkpoints under `./runs/` or another user-defined output directory.
- Review command-line defaults before running large experiments.
- This package does not include author, institution, homepage, or contact information.
