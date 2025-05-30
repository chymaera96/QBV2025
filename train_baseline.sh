export PYTHONPATH=$(pwd)/src
python src/qvim_mn_baseline/ex_qvim.py --num_gpus 1 --precision bf16-mixed \
                                        --batch_size 64 --acc_grad 1 \
                                        --proj_dim 128 \
                                        --n_epochs 100 \
                                        --warmup_steps 0 \
                                        --id passt_2_2

