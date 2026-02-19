

python -m torch.distributed.launch --use-env --nproc_per_node=2 --master_port=29509 3_train_rcg.py



CUDA_VISIBLE_DEVICES=0 python 3_train_cyclegan_3d.py



CUDA_VISIBLE_DEVICES=4 python 3_train_hinet.py

CUDA_VISIBLE_DEVICES=4,5,6,7 python 3_train_i2imamba.py


python 3_train_medsyn.py

CUDA_VISIBLE_DEVICES=4 python 3_train_unest.py
