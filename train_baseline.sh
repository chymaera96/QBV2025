export PYTHONPATH=$(pwd)/src
# python src/qvim_mn_baseline/ex_qvim.py --num_gpus 1 --precision bf16-mixed \
#                                         --batch_size 128 \
#                                         --n_epochs 30 \
#                                         --id mn_2_2

python src/qvim_mn_baseline/pretrain.py --precision bf16-mixed \
                                        --batch_size 128 \
                                        --n_epochs 100 \
                                        --dataset_path /data/EECS-MachineListeningLab/datasets/FSD50K/dev_audio \
                                        --id pretrain_0
