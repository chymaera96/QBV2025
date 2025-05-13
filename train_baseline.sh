export PYTHONPATH=$(pwd)/src
python src/qvim_mn_baseline/ex_qvim.py --num_gpus 2 --precision bf16-mixed --batch_size 128 --acc_grad 1 --n_epochs 100 --tau_trainable --id mbn_mixed_b128_gpu2_ep100_sche_traintau

