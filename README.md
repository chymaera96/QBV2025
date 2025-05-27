# QbV with CLAP
## Query-by-Vocal Imitation Challenge

This is a starter code for CLAP experiments, including only the editable installation of CLAP.

# Install
This installs the baseline and `CLAP` in editable mode.
```
pip install -r requirements.txt
```

When run for the first time (e.g., below), the checkpoint is saved to `./CLAP/src/laion_clap/xxx.pt` and ignored by Git:

```
import laion_clap
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt(model_id=0) # download the pretrained checkpoint.
```

Only model IDs 0–3 are supported by default, but more are available [here](https://huggingface.co/lukewys/laion_clap/tree/main).
You can use them by modifying [`hook.py`](https://github.com/LAION-AI/CLAP/blob/cc8f7654fc8b718434cf9ac6e6faf72f78c2797b/src/laion_clap/hook.py#L75).




## Train
Simply run or modify `train_basline.sh`.


## Performance and Insights

### Frozen CLAP retrieval

CLAP 0-3 use the representations from the pretrained checkpoints, without further training.
* CLAP 0: '630k-best.pt',
* CLAP 1: '630k-audioset-best.pt',
* CLAP 2: '630k-fusion-best.pt',
* CLAP 3: '630k-audioset-fusion-best.pt'
* CLAP AF: CLAP from audio-flamingo 2

| Model Name   | MRR (exact match) | NDCG (category match) |
|--------------|-------------------|-----------------------|
| random       | 0.0444            | ~0.337                |
| 2DFT         | 0.1262            | 0.4793                |
| MN baseline  | 0.2726            | 0.6463                |
| CLAP 0       | 0.1156            | 0.4752                |
| CLAP 1       | 0.1316            | 0.4906                |
| CLAP 2       | 0.1167            | 0.4851                |
| CLAP 3       | 0.1301            | 0.4940                |
| CLAP AF      | 0.1127            | 0.4743                |

### Frozen CLAP retrieval with other layers
