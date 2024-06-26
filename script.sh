#!/bin/bash
python train_predictor.py --loss mae 
python train_predictor.py --loss mae --relu
python train_predictor.py --loss mse 
python train_predictor.py --loss mse --relu
python train_predictor.py --loss both 
python train_predictor.py --loss both --relu