import numpy as np
import torch
print(np.__version__)  # 确认 NumPy 版本
print(torch.from_numpy(np.array([1, 2, 3])))  # 测试 NumPy 是否正常工作
