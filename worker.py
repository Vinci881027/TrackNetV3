import os
import json
import queue
import numpy as np
import multiprocessing as mp
import subprocess as sp
from argparse import ArgumentParser

# 1. How many workers do you want for each GPU?
# ---------------------------------------------
# E.g. you have 2 RTX 2080 (8GB) and 2 RTX 2080 Ti (11GB),
# configure the number of processes to be run in parallel for each GPU.
CUDA_WORKERS = {
    # 0: 1, # 3 workers for CUDA_VISIBLE_DEVICES=0
    # 1: 1, # 3 workers for CUDA_VISIBLE_DEVICES=1
    2: 1, # 4 workers for CUDA_VISIBLE_DEVICES=2
    3: 1, # 4 workers for CUDA_VISIBLE_DEVICES=3
    4: 1, # 4 workers for CUDA_VISIBLE_DEVICES=4 
    5: 1, # 4 workers for CUDA_VISIBLE_DEVICES=5
    6: 1, # 4 workers for CUDA_VISIBLE_DEVICES=6
    7: 1, # 4 workers for CUDA_VISIBLE_DEVICES=7
    8: 1, # 4 workers for CUDA_VISIBLE_DEVICES=8
    9: 1, # 4 workers for CUDA_VISIBLE_DEVICES=9
}

# 2. List of commands you want to run
# ---------------------------------------------
# pick training points by uniform sampling
parser = ArgumentParser(description="Worker script parameters")
parser.add_argument("--match", type=str, required=True, help="Which match")
args = parser.parse_args()
match = args.match
video_dir = sorted(os.listdir(f"{match}"))

COMMANDS = [
    ['python', 'predict.py',
    '--video_file', f'{match}/{video}',
    '--tracknet_file', f'exp/TrackNet_best.pt',
    '--save_dir', f'prediction_{match}',
    '--output_video',
    '--large_video',
    ]
    for video in video_dir if video.endswith('.mp4')
]

def worker(cuda_no, worker_no, cmd_queue):
    worker_name = 'CUDA-{}:{}'.format(cuda_no, worker_no)
    print(worker_name, 'started')
    
    env = os.environ.copy()
    # overwrite visible cuda devices
    env['CUDA_VISIBLE_DEVICES'] = str(cuda_no)#'4, 5'#
    
    while True:
        cmd = cmd_queue.get()
        
        if cmd is None:
            cmd_queue.task_done()
            break
        
        print(worker_name, cmd)
        
        shell = {str: True, list: False}.get(type(cmd))
        assert shell is not None, 'cmd should be list or str'
        
        sp.Popen(cmd, shell=shell, env=env).wait()
        cmd_queue.task_done()
    
    print(worker_name, 'stopped')


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    cmd_queue = mp.JoinableQueue()
    
    for cmd in COMMANDS:
        cmd_queue.put(cmd)
        
    for _ in range(sum(CUDA_WORKERS.values())):
        # workers stop after getting None
        cmd_queue.put(None)
        
    procs = [
        mp.Process(target=worker, args=(cuda_no, worker_no, cmd_queue), daemon=True)
        for cuda_no, num_workers in CUDA_WORKERS.items()
        for worker_no in range(num_workers)
    ]
    
    for proc in procs:
        proc.start()

    cmd_queue.join()
        
    for proc in procs:
        proc.join()