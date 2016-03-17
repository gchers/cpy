#!/usr/local/bin/env python
import ncm_utils
import numpy as np


class KDE(ncm_utils.NCM):
    
    
    def __init__(self, h, kernel):
        """Kernel Density Estimation (KDE) non-conformity measure.
    
        Keyword arguments:
            h: bandwidth
            kernel: kernel function, which can be in ['gaussian',] or can be a function
                calculating the kernel over a vector and returning a scalar
        """
        self.h = h
        if isinstance(kernel, str):
            if kernel == 'gaussian':
                 self.kernel = self.__gaussian_kernel
            else:
                raise Exception('Non recognized kernel function')
        elif isinstance(kernel, function):
            self.kernel = kernel
        else:
            raise Exception("`kernel' must either be a string or a kernel function")
        
    def compute(self, z, Z):
        """Return Kernel Density Estimation (KDE) non-conformity measure.
    
        Keyword arguments:
            z: numpy array
               The example on which to calculate the measure
            Z: two dimensional numpy array
               Contains examples one per row, excluding z.
        """
        z_tmp = np.vstack((Z, z))
        (N, d) = z_tmp.shape
        r = 0.0
        for zi in z_tmp:
            r -= self.kernel((z - zi) / self.h)
        r /= N * (self.h ** d)
    
        return r
    
    def __gaussian_kernel(self, z):
        """Gaussian kernel.
        
        Keyword arguments:
            z: vector
        """
        return np.exp(-0.5 * np.dot(z, z)) / np.sqrt(2 * np.pi)
