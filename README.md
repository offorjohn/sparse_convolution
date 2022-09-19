# sparse_convolution
Sparse convolution in python

Install:
`git clone https://github.com/RichieHakim/sparse_convolution` \
`cd sparse_convolution` \
`pip install -e .` \

Basic usage:
```
import sparse_convolution as sc
import numpy as np
import scipy.sparse

# Create a single sparse matrix
A = scipy.sparse.rand(100, 100, density=0.1)

# Create a dense kernel
B = np.random.rand(3, 3)

# Prepare class
conv = Toeplitz_convolution2d(
    x_shape=A.shape,
    k=B,
    mode='same',
    dtype=np.float32,
)

# Convolve
C = conv(
    x=A,
    batching=False,
    mode='same',
)
```