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

.
