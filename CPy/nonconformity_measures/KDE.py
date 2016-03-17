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
        
    def compute(self, zn, z):
        """Return Kernel Density Estimation (KDE) non-conformity measure.
    
        Keyword arguments:
            zn the example on which to calculate the measure
            z: numpy.array containing examples one per row, excluding zn
        """
        z_tmp = np.vstack((z, zn))
        (N, d) = z_tmp.shape
        r = 0.0
        for zi in z_tmp:
            r -= self.kernel((zn - zi) / self.h)
        r /= N * (self.h ** d)
    
        return r
    
    def __gaussian_kernel(self, u):
        """Gaussian kernel.
        
        Keyword arguments:
            u: vector
        """
        return np.exp(-0.5 * np.dot(u, u)) / np.sqrt(2 * np.pi)
