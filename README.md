# PSPD
Run :

Train Teacher model:
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
                        --nproc_per_node=1 \
                        --master_port 51321 \
                        run.py \
                        -b 64 \
                        --epochs 300 \
                        --lr 0.0003 \
                        -p 50  \
                        --arch sfcn \
                        --data /path/to/data \
                        --T 10
```  
Train Student model by adding teacher path (-t)
```shell
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
                        --nproc_per_node=1 \
                        --master_port 51321 \
                        run.py \
                        -b 64 \
                        --epochs 300 \
                        --lr 0.0003 \
                        -p 50  \
                        --arch sfcn \
                        --data /path/to/data \
                        --T 10 \
                        -t /path/to/model
```
