from typing import Tuple, Optional, Union

import scipy.sparse
import numpy as np

class Toeplitz_convolution2d():
    """
    Convolve a 2D array with a 2D kernel using the Toeplitz matrix
    multiplication method. This class is ideal when 'x' is very sparse
    (density<0.01), 'x' is small (shape <(1000,1000)), 'k' is small (shape
    <(100,100)), and the batch size is large (e.g. 1000+). Generally, it is
    faster than scipy.signal.convolve2d when convolving multiple arrays with the
    same kernel. It maintains a low memory footprint by storing the toeplitz
    matrix as a sparse matrix.
    RH 2022

    Attributes:
        x_shape (Tuple[int, int]):
            The shape of the 2D array to be convolved.
        k (np.ndarray):
            2D kernel to convolve with.
        mode (str):
            Either ``'full'``, ``'same'``, or ``'valid'``. See
            scipy.signal.convolve2d for details.
        dtype (Optional[np.dtype]):
            The data type to use for the Toeplitz matrix.
            If ``None``, then the data type of the kernel is used.

    Args:
        x_shape (Tuple[int, int]):
            The shape of the 2D array to be convolved.
        k (np.ndarray):
            2D kernel to convolve with.
        mode (str):
            Convolution method to use, either ``'full'``, ``'same'``, or
            ``'valid'``.
            See scipy.signal.convolve2d for details. (Default is 'same')
        dtype (Optional[np.dtype]):
            The data type to use for the Toeplitz matrix. Ideally, this matches
            the data type of the input array. If ``None``, then the data type of
            the kernel is used. (Default is ``None``)

    Example:
        .. highlight:: python
        .. code-block:: python

            # create Toeplitz_convolution2d object
            toeplitz_convolution2d = Toeplitz_convolution2d(
                x_shape=(100,30),
                k=np.random.rand(10,10),
                mode='same',
            )
            toeplitz_convolution2d(
                x=scipy.sparse.csr_matrix(np.random.rand(5,3000)),
                batch_size=True,
            )
    """
    def __init__(
        self,
        x_shape: Tuple[int, int],
        k: np.ndarray,
        mode: str = 'same',
        dtype: Optional[np.dtype] = None,
    ):
        """
        Initializes the Toeplitz_convolution2d object and stores the Toeplitz
        matrix.
        """
        self.k = k = np.flipud(k.copy())
        self.mode = mode
        self.x_shape = x_shape
        dtype = k.dtype if dtype is None else dtype

        if mode == 'valid':
            assert x_shape[0] >= k.shape[0] and x_shape[1] >= k.shape[1], "x must be larger than k in both dimensions for mode='valid'"

        self.so = so = size_output_array = ( (k.shape[0] + x_shape[0] -1), (k.shape[1] + x_shape[1] -1))  ## 'size out' is the size of the output array

        ## make the toeplitz matrices
        t = toeplitz_matrices = [scipy.sparse.diags(
            diagonals=np.ones((k.shape[1], x_shape[1]), dtype=dtype) * k_i[::-1][:,None], 
            offsets=np.arange(-k.shape[1]+1, 1), 
            shape=(so[1], x_shape[1]),
            dtype=dtype,
        ) for k_i in k[::-1]]  ## make the toeplitz matrices for the rows of the kernel
        tc = toeplitz_concatenated = scipy.sparse.vstack(t + [scipy.sparse.dia_matrix((t[0].shape), dtype=dtype)]*(x_shape[0]-1))  ## add empty matrices to the bottom of the block due to padding, then concatenate

        ## make the double block toeplitz matrix
        self.dt = double_toeplitz = scipy.sparse.hstack([self._roll_sparse(
            x=tc, 
            shift=(ii>0)*ii*(so[1])  ## shift the blocks by the size of the output array
        ) for ii in range(x_shape[0])]).tocsr()
    
    def __call__(
        self,
        x: Union[np.ndarray, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix],
        batching: bool = True,
        mode: Optional[str] = None,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """
        Convolve the input array with the kernel.

        Args:
            x (Union[np.ndarray, scipy.sparse.csc_matrix,
            scipy.sparse.csr_matrix]): 
                Input array(s) (i.e. image(s)) to convolve with the kernel. \n
                * If ``batching==False``: Single 2D array to convolve with the
                  kernel. Shape: *(self.x_shape[0], self.x_shape[1])*
                * If ``batching==True``: Multiple 2D arrays that have been
                  flattened into row vectors (with order='C'). \n
                Shape: *(n_arrays, self.x_shape[0]*self.x_shape[1])*

            batching (bool): 
                * ``False``: x is a single 2D array.
                * ``True``: x is a 2D array where each row is a flattened 2D
                  array. \n
                (Default is ``True``)

            mode (Optional[str]): 
                Defines the mode of the convolution. Options are 'full', 'same'
                or 'valid'. See `scipy.signal.convolve2d` for details. Overrides
                the mode set in __init__. (Default is ``None``)

        Returns:
            (Union[np.ndarray, scipy.sparse.csr_matrix]):
                out (Union[np.ndarray, scipy.sparse.csr_matrix]): 
                    * ``batching==True``: Multiple convolved 2D arrays that have
                      been flattened into row vectors (with order='C'). Shape:
                      *(n_arrays, height*width)*
                    * ``batching==False``: Single convolved 2D array of shape
                      *(height, width)*
        """
        if mode is None:
            mode = self.mode  ## use the mode that was set in the init if not specified
        issparse = scipy.sparse.issparse(x)
        
        if batching:
            x_v = x.T  ## transpose into column vectors
        else:
            x_v = x.reshape(-1, 1)  ## reshape 2D array into a column vector
        
        if issparse:
            x_v = x_v.tocsc()
        
        out_v = self.dt @ x_v  ## if sparse, then 'out_v' will be a csc matrix
            
        ## crop the output to the correct size
        if mode == 'full':
            p_t = 0
            p_b = self.so[0]+1
            p_l = 0
            p_r = self.so[1]+1
        if mode == 'same':
            p_t = (self.k.shape[0]-1)//2
            p_b = -(self.k.shape[0]-1)//2
            p_l = (self.k.shape[1]-1)//2
            p_r = -(self.k.shape[1]-1)//2

            p_b = self.x_shape[0]+1 if p_b==0 else p_b
            p_r = self.x_shape[1]+1 if p_r==0 else p_r
        if mode == 'valid':
            p_t = (self.k.shape[0]-1)
            p_b = -(self.k.shape[0]-1)
            p_l = (self.k.shape[1]-1)
            p_r = -(self.k.shape[1]-1)

            p_b = self.x_shape[0]+1 if p_b==0 else p_b
            p_r = self.x_shape[1]+1 if p_r==0 else p_r
        
        if batching:
            idx_crop = np.zeros((self.so), dtype=np.bool_)
            idx_crop[p_t:p_b, p_l:p_r] = True
            idx_crop = idx_crop.reshape(-1)
            out = out_v[idx_crop,:].T
        else:
            if issparse:
                out = out_v.reshape((self.so)).tocsc()[p_t:p_b, p_l:p_r]
            else:
                out = out_v.reshape((self.so))[p_t:p_b, p_l:p_r]  ## reshape back into 2D array and crop
        return out
    
    def _roll_sparse(
        self,
        x: scipy.sparse.csr_matrix,
        shift: int,
    ):
        """
        Roll columns of a sparse matrix.
        """
        out = x.copy()
        out.row += shift
        return out