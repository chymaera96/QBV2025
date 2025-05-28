export PYTHONPATH=$(pwd)/src
python src/qvim_mn_baseline/ex_qvim.py --num_gpus 1 --precision bf16-mixed \
                                        --batch_size 128 --acc_grad 1 \
                                        --n_epochs 100 \
                                        --loss_ratio 0.2 \
                                        --mask_ratio 0.6 \
                                        --id jepa_4

