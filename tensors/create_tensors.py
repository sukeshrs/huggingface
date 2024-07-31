import torch
import numpy as np

# 1. tensor from a list
data = [[1,2], [3,4]]
x_data = torch.tensor(data)
print(x_data)


# 2. Create a numpy array from data
np_array = np.array(data)
print(np_array) # prints the numpy array
x_np = torch.from_numpy(np_array)
print(x_np)


# 3. From another tensor
x_ones = torch.ones_like(x_data)
print(x_ones)
# overrides the dtype to float
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)

# 4. From constat values
shape = (2, 3)  # a tuple defining the shape of the tensor
random_tensor = torch.rand(shape)
print(random_tensor)
ones_tensor = torch.ones(shape)
print("Ones: ",ones_tensor)
zeros_tensor = torch.zeros(shape)
print("Zeroes: " , zeros_tensor)


