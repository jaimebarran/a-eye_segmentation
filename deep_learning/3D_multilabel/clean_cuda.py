import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    # device = cuda.get_current_device()
    # device.reset()

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()