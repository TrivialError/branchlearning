import torch
def check_cuda():
    return False
    if torch.cuda.is_available():
        return True
    return False