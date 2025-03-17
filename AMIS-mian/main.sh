export OMP_NUM_THREADS=16
python -m torch.distributed.run --nproc_per_node=2 --master_port=29500 main.py --config ./configs/voc.yaml