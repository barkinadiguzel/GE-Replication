import torch.nn.functional as F

def global_avg(x):
    return F.adaptive_avg_pool2d(x, 1)


def global_max(x):
    return F.adaptive_max_pool2d(x, 1)
