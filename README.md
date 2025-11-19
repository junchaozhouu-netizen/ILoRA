ILoRA: Federated Learning with Low-Rank Adaptation for Heterogeneous Client Aggregation
======================================================================================

Overview
--------

This repository contains executable scripts, datasets, and configuration templates for reproducing the ILoRA study on low-rank adaptation under heterogeneous client aggregation. The codebase is organized into three main stacks—computer vision (CV), natural language processing (NLP), and ablation studies—that share a common Conda environment and logging strategy.

Environment Setup
-----------------

All experiments were carried out inside the `LORA+SAM5_cifar10` Conda environment on Python 3.9.6 and CUDA 12.1. Recreate the environment with the exact package sequence that was used to generate the reported results:

```
conda create --name LORA+SAM5_cifar10 python=3.9.6
conda activate LORA+SAM5_cifar10

pip install flwr==1.10.0
pip install ray==2.6.3
pip install flwr-datasets[vision]==0.2.0
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install matplotlib==3.8.3
pip install scikit-learn==1.4.2
pip install seaborn==0.13.2
pip install ipywidgets==8.1.2

pip install transformers==4.37.2
pip install accelerate==0.30.0
conda install jupyter notebook
pip install "numpy<2"
pip install chardet

pip install peft==0.13.2 --no-dependencies
# A blank 'pip install transformers==' was logged during experimentation; keep the pinned 4.37.2 wheel above to avoid resolver issues.
pip install accelerate==1.0.1
pip install datasets==2.19.2
pip install timm==1.0.12
pip install imageio==2.35.1
pip install tensorboardX

# Newly added utilities
pip install charset_normalizer
pip install transformers
pip install chardet charset-normalizer
pip install hf_xet
pip install tensorboard

python -m tensorboard.main --logdir ./runs
```

> Tip: Keep `pip install transformers` after the pinned 4.37.2 installation only if you need the latest nightly wheel; otherwise skip it to stay on the verified dependency set.

Configuration Files
-------------------

Three YAML files in `configs/` describe every experiment group end-to-end:

- `ablation_config.yaml` — documents the client-drift, large-heterogeneity, alpha-sensitivity, and component-growth studies. Each entry specifies the script path, dataset, LoRA rank layout, optimizer choice, and sweep parameters.
- `cv_config.yaml` — covers CIFAR-10, CIFAR-100, DomainNet, and large-scale multi-client FL jobs. It lists model backbones (ViT/Swin), heterogeneity knobs, logging directories, and optimizer hooks such as SAM.
- `nlp_config.yaml` — enumerates AG News, DBPedia, IMDB, and HateXplain scenarios, highlighting tokenizer limits, control variate toggles, and per-task LoRA rank overrides.

You can keep these files purely as documentation or feed them to your orchestration layer to autogenerate launch commands.

Running Experiments
-------------------

All launch scripts live under `scripts/`:

- CV: `scripts/cv/run_base_cv.py`, `scripts/cv/run_domainnet.py`, `scripts/cv/run_fl_multi_client.py`
- NLP: `scripts/nlp/run_nlp_adamw_0_5.py`, `scripts/nlp/run_nlp_adamw_0_6.py`
- Ablation: files under `scripts/ablation/`

Example (AG News AdamW, alpha 0.5):

```
cd id4914_code
conda activate LORA+SAM5_cifar10
python scripts/nlp/run_nlp_adamw_0_5.py ^
  --dataset AG_NEWS ^
  --num_clients 6 ^
  --alpha 0.5 ^
  --heterogeneous_rank_clients 2,4,4,6,6,8 ^
  --save_dir ./runs/nlp/agnews_adamw05
```

Customize the arguments according to the YAML templates to keep your experimental notes synchronized with the actual runs.

Logging and Monitoring
----------------------

- Default checkpoint root: `./runs/`
- TensorBoard (covers CV, NLP, and ablation logs): `python -m tensorboard.main --logdir ./runs`

Make sure `tensorboardX` and `tensorboard` are installed inside the Conda environment before streaming logs.

Reproducing Results
-------------------

1. Follow the Environment Setup section exactly.
2. Download datasets into `src/dataset/` (see the individual scripts for download logic).
3. Pick the relevant YAML entry and translate it into the corresponding script arguments.
4. Launch TensorBoard to monitor training curves across federated rounds.
5. Archive `./runs/` for every experiment to keep the aggregation statistics and checkpoints aligned with the configuration you used.

