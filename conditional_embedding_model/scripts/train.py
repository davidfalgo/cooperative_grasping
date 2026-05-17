#!/usr/bin/env python3
# Reorganized from train/TrainNS_train.py
# Original author:
# Description: Minimal entrypoint that launches a W&B hyperparameter sweep via trainer.py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from conditional_embedding_model.training import sweep

if __name__ == '__main__':
    sweep()
