#!/usr/local/bin/env python
import ncm_utils
import numpy as np


class KDE(ncm_utils.NCM):
    
    
    def __init__(self, h, kernel):
        """Kernel Density Estimation (KDE) non-conformity measure.
    
        Parameters
        ----------
        h : float
            Bandwidth for KDE.
        kernel : string or function
            Kernel function for KDE. It can be a string in ['gaussian'], which indicates
            a gaussian kernel, or a function. In the second case, the function should
            describe a kernel function, and must:
            accept an array-like vector of shape (n_features,), and return a float.
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
        """Return the Kernel Density Estimation (KDE) non-conformity measure for
        vector z.
    
        Parameters
        ----------
        z : array-like, shape (n_features,)
            Test vector, where n_features is the number of features.
        Z : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples,
            n_features is the number of features.

        Returns
        -------
        r : float
            KDE nonconformity measure on vector z with respect to Z.
        """
        z_tmp = np.vstack((Z, z))
        (N, d) = z_tmp.shape
        r = 0.0
        for zi in z_tmp:
            r -= self.kernel((z - zi) / self.h)
        r /= N * (self.h ** d)
    
        return r
    
    def __gaussian_kernel(self, u):
        """Gaussian kernel.
        
        Parameters
        ----------
        u : array-like, shape (n_features,)
            Vector, where n_features is the number of features.

        Returns
        -------
        s : float
            Gaussian kernel over vector u.
        """
        s = np.exp(-0.5 * np.dot(u, u)) / np.sqrt(2 * np.pi)

        return s
