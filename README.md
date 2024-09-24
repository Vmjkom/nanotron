<h1 align="center">⚡️ Nanotron</h1>

<p align="center">
    <a href="https://github.com/huggingface/nanotron/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/nanotron.svg">
    </a>
    <a href="https://github.com/huggingface/nanotron/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/huggingface/nanotron.svg?color=green">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="#installation">Installation</a> •
        <a href="#quick-start">Quick Start</a> •
        <a href="#features">Features</a> •
        <a href="CONTRIBUTING.md">Contributing</a>
    <p>
</h4>

<h3 align="center">
    <a href="https://huggingface.co/nanotron"><img style="float: middle; padding: 10px 10px 10px 10px;" width="60" height="55" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" /></a>
</h3>
<h3 align="center">
<p>Pretraining models made easy
</h3>


Nanotron is a library for pretraining transformer models. It provides a simple and flexible API to pretrain models on custom datasets. Nanotron is designed to be easy to use, fast, and scalable. It is built with the following principles in mind:

- **Simplicity**: Nanotron is designed to be easy to use. It provides a simple and flexible API to pretrain models on custom datasets.
- **Performance**: Optimized for speed and scalability, Nanotron uses the latest techniques to train models faster and more efficiently.

# LUMI
## Setup
You can do the following on a login node, as all of the gpu related installations are arleady in the module/contaer we use
```bash
module purge

# Get access to the csc provided modules
module use /appl/local/csc/modulefiles #Consider adding this to your .bashrc or .profile
module load pytorch/2.4 #As of 24.9.2024 the latest is this. The previous versions propably wont work

#Right now we use a naughty virtual enviroment, but expect this to change to a fully containerized enviroment

python3 -m venv .venv --system-site-packages 
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[nanosets]
```
## Data
If your dataset is available in huggingface format you set it in your .yaml config file like so:
```yaml
data_stages:
- data:
    dataset:
      dataset_overwrite_cache: false
      dataset_processing_num_proc_per_process: 7
      hf_dataset_config_name: null
      hf_dataset_or_datasets:
          roneneldan/TinyStories: 0.5
      hf_dataset_splits: train
      text_column_name: text
    num_loading_workers: 0
    seed: 42
  name: Stable Training Stage
  start_training_step: 1

```
### Preprocess
Larger datasets can be preprocessed with [`/tools/preprocess_data.py`](/tools/preprocess_data.py). This is a script that read in and process a large dataset in various ways, in a parallel fashion. This is done with the [`datatrove library`](https://github.com/huggingface/datatrove) 

These preprocessed datasets are called "nanosets" and are configured in the yaml file a little differently:
```yaml
data_stages:
  - data:
      dataset:
        dataset_folder: /scratch/project_462000353/data/nanosets/fineweb-edu/350BT
      num_loading_workers: 7
      seed: 42
    name: Stable Training Stage
    start_training_step: 1

```
More info for these is in [`/tools/nanoset.md`](/tools/nanoset.md).

See all of the dataset related configuration parameters in [`config.py`](/src/nanotron/config/config.py)

## Fineweb ablations
If your wish is to do pretraining for a fineweb-like ablation study, you can follow these steps:
Modify the [`llama_2B.yaml`](/configs/llama_2B.yaml)config file to point to your own datasets, directories for checkpoints etc.
The model parameters should be left untouched if you want to replicate the 1.82B llama model huggingface used.
Modify [`slurm_script`](/slurm_scripts/train.sh) and add your config file ``export CONFIG=$DIR/configs/llama_2B.yaml``
```bash
sbatch /slurm_scripts/train.sh

#Or for quick debugging launch an interactive session with salloc
#PARAMS: 2 nodes, 30 minutes run time, job name
./slurm_scripts/interactive.sh 2 00:30:00 debug-nanotron

#And then to launch after your resources have been allocated
./slurm_scripts/train.sh
```
## TODO
- [ ] Implement [`lighteval`](/src/nanotron/config/lighteval_config.py) into the pretraining
- [ ] Others?
# End of LUMI spesific README
> [!TIP]
> We log to wandb automatically if it's installed. For that you can use `pip install wandb`. If you don't want to use wandb, you can run `wandb disabled`.

## Quick Start
### Training a tiny Llama model
The following command will train a tiny Llama model on a single node with 8 GPUs. The model will be saved in the `checkpoints` directory as specified in the config file.
```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```

### Run generation from your checkpoint
```bash
torchrun --nproc_per_node=1 run_generate.py --ckpt-path checkpoints/10/ --tp 1 --pp 1
# We could set a larger TP for faster generation, and a larger PP in case of very large models.
```

### Custom examples
You can find more examples in the [`/examples`](/examples) directory:
<!-- Make a table of the examples we support -->
| Example | Description |
| --- | --- |
| `custom-dataloader` | Plug a custom dataloader to nanotron |
| `datatrove` | Use the datatrove library to load data |
| `doremi` | Use DoReMi to speed up training |
| `mamba` | Train an example Mamba model |
| `moe` | Train an example Mixture-of-Experts (MoE) model |
| `mup` | Use spectral µTransfer to scale up your model |
| `examples/config_tiny_llama_with_s3_upload.yaml` | For automatically uploading checkpoints to S3 |

We're working on adding more examples soon! Feel free to add a PR to add your own example. 🚀


## Features
We currently support the following features:
- [x] 3D parallelism (DP+TP+PP)
- [x] Expert parallelism for MoEs
- [x] AFAB and 1F1B schedules for PP
- [x] Explicit APIs for TP and PP which enables easy debugging
- [x] ZeRO-1 optimizer
- [x] FP32 gradient accumulation
- [x] Parameter tying/sharding
- [x] Custom module checkpointing for large models
- [x] Spectral µTransfer parametrization for scaling up neural networks
- [x] Mamba example

And we have on our roadmap:
- [ ] FP8 training
- [ ] ZeRO-3 optimizer (a.k.a FSDP)
- [ ] `torch.compile` support
- [ ] Ring attention
- [ ] Interleaved 1f1b schedule

## Credits
We would like to thank everyone working on LLMs, especially those sharing their work openly from which we took great inspiration: Nvidia for `Megatron-LM/apex`, Microsoft for `DeepSpeed`, HazyResearch for `flash-attn`..
