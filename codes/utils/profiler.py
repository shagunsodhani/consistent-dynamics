import torch
import subprocess
import os

def get_cuda_memory_allocated():
    # In Mb
    return torch.cuda.memory_allocated() / (2 ** 20)

def get_cuda_memory_cached():
    # In Mb
    return torch.cuda.memory_cached()/(2**20)

def get_gpu_memory_map():
    """Get the current gpu usage. Taken from https://gist.github.com/vardaan123/53a49a789b27bf829bb3799c60e26705

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().decode('utf-8').split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print(gpu_memory_map)
    return gpu_memory_map


def get_cpu_memory_map():
    '''Get the cpu usage of the current process'''
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 20  # memory use in GB...I think
    print('memory use:', memory_use)
    return memory_use
