# sparse_convolution
Sparse convolution in python

Install: \
`git clone https://github.com/RichieHakim/sparse_convolution` \
`cd sparse_convolution` \
`pip install -e .` 


Basic usage: \
Convolve a single sparse 2D array with a 2D kernel.
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
).toarray()
```


Batching usage: \
Convolve multiple sparse 2D arrays with a 2D kernel. \
The input arrays must be reshaped into flattened vectors and stacked into a single sparse array of shape: `(n_arrays, height * width)`. 
```
import sparse_convolution as sc
import numpy as np
import scipy.sparse

# Create multiple sparse matrices
# note that the shape of A will be (3, 100**2)
A = scipy.sparse.vstack([
    scipy.sparse.rand(100, 100, density=0.1).reshape(1, -1),
    scipy.sparse.rand(100, 100, density=0.1).reshape(1, -1),
    scipy.sparse.rand(100, 100, density=0.1).reshape(1, -1),
]).tocsr()

# Create a dense kernel
B = np.random.rand(3, 3)

# Prepare class
conv = sc.Toeplitz_convolution2d(
    x_shape=(100, 100),  # note that the input shape here is (100, 100)
    k=B,
    mode='same',
    dtype=np.float32,
)

# Convolve
C = conv(
    x=A,
    batching=True,
    mode='same',
)

# Reshape the output back to (3, 100, 100)
C_reshaped = np.stack([c.reshape(100, 100).toarray() for c in C], axis=0)
```