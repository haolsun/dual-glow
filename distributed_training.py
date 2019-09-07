import multiprocessing
import subprocess
import os

max_folds = 71

def get_next_fold(lock):
    fold_idx = 0
    yield fold_idx
    fold_idx += 1

def get_next_Popen_args(lock):
    return ['python' 'train.py' '--problem' 'asl2pet' '--att' 'year_diff' '--pmap' '28' \
    '--n_batch_train' '1' '--n_batch_test' '1' '--n_visual_row' '2' '--n_batch_init' '1' '--lr' '0.0001' \
    '--epochs' '300' '--epochs_warmup' '10' '--epochs_full_valid' '10' '--depth' '2' '4' '8' '6' '--ycond' \
    '--seed 806' '--n_l 4' '--fold'].append(str(get_next_fold(lock)))

class GPUJob:
    def __init__(self, PopenArgs):
        self.args = PopenArgs
        self.proc = []

    def start():
        

class GPUWorker:
    def __init__(self, gpu_id, lock):
        self.environ = os.environ.copy()
        self.environ['CUDA_VISIBLE_DEVICE'] = gpu_id
        self.lock = lock
    
    def start(self):