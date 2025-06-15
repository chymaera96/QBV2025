export PYTHONPATH=$(pwd)/src


# python src/qvim_mn_baseline/pretrain.py --precision bf16-mixed \
#                                         --batch_size 128 \
#                                         --n_epochs 100 \
#                                         --dataset_path /data/EECS-MachineListeningLab/datasets/FSD50K/dev_audio \
#                                         --id pretrain_1

python src/qvim_mn_baseline/ex_qvim.py --precision bf16-mixed \
                                        --batch_size 128 \
                                        --n_epochs 20 \
                                        --id sht0
