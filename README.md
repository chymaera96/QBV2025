# AES AIMLA Challenge 2025 Baseline System
## Query-by-Vocal Imitation Challenge

This is a modified version of the official baseline for the QVIM challenge.
Please read the description below for more details.

## Modifications
* Supports DDP (Distributed Data Parallel) for multi-GPU training
* Enables mixed-precision training (with no performance degradation)
* Allows control of the cosine scheduler using percentages
* New arguments:

  * `--precision`: Precision mode `{32, 16, bf16, bf16-mixed}`
  * `--warmup_percent`: Scheduler warm-up phase (e.g., default = 0.1 → 10%)
  * `--rampdown_percent`: Cosine ramp-down phase (e.g., default = 0.8 → 80%)
  * `--sync_batchnorm`: Based on testing, may lead to performance drop and slower training
  * `--acc_grad`: Number of gradient accumulation steps (default = 1)
  * `--id`: WandB run ID. If not specified, a random ID will be generated

# Install
```
pip install -r requirements.txt
```

## Train
Simply run or modify bash train_basline.sh.
The dataset will be downloaded automatically on the first run.
The following command gives the best results:
```
export PYTHONPATH=$(pwd)/src
python src/qvim_mn_baseline/ex_qvim.py \
  --num_gpus 2 \
  --precision bf16-mixed \
  --batch_size 128 \
  --acc_grad 1 \
  --n_epochs 15 \
  --id mbn_mixed_b128_gpu2
```

## Performance and Insights
<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
    <a href="https://github.com/user-attachments/assets/3e10bc45-4c20-425a-a3bc-cf38ff812780" target="_blank">
        <img width="450" alt="Screenshot 2025-05-13 at 12 30 19" src="https://github.com/user-attachments/assets/3e10bc45-4c20-425a-a3bc-cf38ff812780" />
    </a>
    <a href="https://github.com/user-attachments/assets/f77ab283-4338-4790-8b68-0bb31ba61934" target="_blank">
        <img width="450" alt="Screenshot 2025-05-13 at 12 30 46" src="https://github.com/user-attachments/assets/f77ab283-4338-4790-8b68-0bb31ba61934" />
    </a>
</div>

<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
    <a href="https://github.com/user-attachments/assets/c4a49a61-98eb-4615-988c-0c1964b3d538" target="_blank">
        <img width="450" alt="Screenshot 2025-05-13 at 12 31 24" src="https://github.com/user-attachments/assets/c4a49a61-98eb-4615-988c-0c1964b3d538" />
    </a>
    <a href="https://github.com/user-attachments/assets/ddeefae0-3250-491f-bfaf-a3751eaba63a" target="_blank">
        <img width="450" alt="Screenshot 2025-05-13 at 12 31 39" src="https://github.com/user-attachments/assets/ddeefae0-3250-491f-bfaf-a3751eaba63a" />
    </a>
</div>

For more details, please refer to the [WandB panel](https://wandb.ai/mimbres/qvim?nw=nwusermimbres).

In terms of MRR:
- Training beyond 15 epochs or with batch size >128 shows no benefit.
- Easy overfitting suggests a need for augmentation.

(end of modification)
---

*Query by Vocal Imitation* (QVIM) enables users to search a database of sounds via a vocal imitation of the desired sound.
This offers sound designers an intuitively expressive way of navigating large sound effects databases. 


We invite participants to submit systems that accept a vocal imitation query and retrieve a perceptually similar recording from a large database of sound effects.

**Important Dates**
- Challenge start: April 1, 2025 
- Challenge end: June 15, 2025
- Challenge results announcement: July 15, 2025

For more details, please have a look at our [website](https://qvim-aes.github.io/#portfolio).

**For Updates** please register [here](https://qvim-aes.github.io/#Registration).

## Baseline System
This repository contains the baseline system for the AES AIMLA Challenge 2025. 
The architecture and the training procedure is based on ["Improving Query-by-Vocal Imitation with Contrastive Learning and Audio Pretraining"](https://dcase.community/documents/workshop2024/proceedings/DCASE2024Workshop_Greif_36.pdf) (DCASE2025 Workshop). 

* The training loop is implemented using [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/). 
* Logging is implemented using [Weights and Biases](https://wandb.ai/site).
* It uses the [MobileNetV3](https://arxiv.org/abs/2211.04772) (MN) pretrained on AudioSet to encode audio recordings.
* The system is trained on [VimSketch](https://interactiveaudiolab.github.io/resources/datasets/vimsketch.html) and evaluated on the public evaluation dataset described on our [website](https://qvim-aes.github.io/#portfolio).


## Getting Started

Prerequisites
- linux (tested on Ubuntu 24.04)
- [conda](https://www.anaconda.com/docs/getting-started/miniconda/install), e.g., [Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)


1. Clone this repository.

```
git clone https://github.com/qvim-aes/qvim-baseline.git
```

2. Create and activate a conda environment with Python 3.10:
```
conda create -n environment.yml
conda activate qvim-baseline
```

3. Install 7z, e.g., 

```
# (on linux)
sudo apt install p7zip-full
# (on windows)
conda install -c conda-forge 7zip
```
*For linux users*: do not use conda package p7zip - this package is based on the outdated version 16.02 of 7zip; to extract the dataset, you need a more recent version of p7zip.

4. If you have not used [Weights and Biases](https://wandb.ai/site) for logging before, you can create a free account. On your
machine, run ```wandb login``` and copy your API key from [this](https://wandb.ai/authorize) link to the command line.


## Training

To start the training, run the following command.
```
cd MAIN_FOLDER_OF_THIS_REPOSITORY
export PYTHONPATH=$(pwd)/src
python src/qvim_mn_baseline/ex_qvim.py
```

## Evaluation Results


| Model Name   | MRR (exact match) | NDCG (category match) |
|--------------|-------------------|-----------------------|
| random       | 0.0444            | ~0.337                |
| 2DFT         | 0.1262            | 0.4793                |
| MN baseline  | 0.2726            | 0.6463                |

- The Mean Reciprocal Rank (MRR) is the metric used to select submitted systems for the subjective evaluation. The MRR gives the average inverse rank $\frac{1}{r_i}$ of the reference sound $i$ averaged over all imitations $Q$:

$$\textrm{MRR} = \frac{1}{\lvert Q \rvert} \sum_{i=1}^{\lvert Q \rvert} \frac{1}{r_i}$$

- The [Normalized Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) (NDCG) measures the systems' ability to retrieve sounds of the imitated category (i.e., how good is the system at retrieving an arbitrary dog bark if a specific dog bark was imitated). 
The NDCG will *not* be used for ranking.

## Contact
For questions or inquiries, please contact [paul.primus@jku.at](mailto:paul.primus@jku.at).


## Citation

```
@inproceedings{Greif2024,
    author = "Greif, Jonathan and Schmid, Florian and Primus, Paul and Widmer, Gerhard",
    title = "Improving Query-By-Vocal Imitation with Contrastive Learning and Audio Pretraining",
    booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2024 Workshop (DCASE2024)",
    address = "Tokyo, Japan",
    month = "October",
    year = "2024",
    pages = "51--55"
}
```
