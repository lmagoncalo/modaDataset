#!/bin/bash
python finetune.py --loss mae --lr 1e-4
python finetune.py --loss mae --relu --lr 1e-4
python finetune.py --loss mse  --lr 1e-4
python finetune.py --loss mse --relu --lr 1e-4
python finetune.py --loss both  --lr 1e-4
python finetune.py --loss both --relu --lr 1e-4

python finetune.py --loss mae --lr 1e-5
python finetune.py --loss mae --relu --lr 1e-5
python finetune.py --loss mse  --lr 1e-5
python finetune.py --loss mse --relu --lr 1e-5
python finetune.py --loss both  --lr 1e-5
python finetune.py --loss both --relu --lr 1e-5