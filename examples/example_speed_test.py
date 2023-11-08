import time

import numpy as np
import torch

SIZE = 100

t1 = torch.randn(SIZE, SIZE, device="cuda")
t2 = torch.randn(SIZE, SIZE, device="cuda")

s = time.time()
t3 = torch.matmul(t1, t2)
print(time.time() - s)

# test numpy

n1 = np.random.randn(SIZE, SIZE)
n2 = np.random.randn(SIZE, SIZE)

s = time.time()

n3 = np.matmul(n1, n2)

print(time.time() - s)
