export PYTHONPATH=$(pwd)/src
python src/qvim_mn_baseline/ex_qvim.py --num_gpus 1 --precision bf16-mixed \
                                        --batch_size 128 --acc_grad 1 \
                                        --n_epochs 100 \
                                        --id passt_2

