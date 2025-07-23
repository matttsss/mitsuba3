import os
import struct

import numpy as np
import matplotlib.pyplot as plt

def read_file(path):
    file_size = os.path.getsize(path)
    res = [[], [], []]
    with open(path, "rb") as file:
        for i in range(file_size // 4):
            res[i%3].append(struct.unpack(f'f', file.read(4))[0])
    return res


cuda_path = "results/time_test_cuda.txt"
cuda_colors = np.array(read_file(cuda_path))

scalar_path = "results/time_test_scalar.txt"
scalar_colors = np.array(read_file(scalar_path))

diff = cuda_colors - scalar_colors

_, offensive_idx = (diff > 0.5).nonzero()
print(cuda_colors[:, offensive_idx])
print(scalar_colors[:, offensive_idx])

plt.plot(diff[0], 'r.')
plt.plot(diff[1], 'g.')
plt.plot(diff[2], 'b.')
plt.show()