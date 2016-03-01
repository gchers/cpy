#!/usr/local/bin/env python
"""Prediction with confidence.

Implementation of prediction with confidence [1]. The implemented non-conformity
measures so far are kNN and KDE (gaussian kernel).

Ref:                                                                         
[1] Shafer, Glenn, and Vladimir Vovk. "A tutorial on conformal prediction."  
The Journal of Machine Learning Research 9 (2008): 371-421.                  
"""
__author__ = "Giovanni Cherubin <g.chers :at: gmail.com>"


def ncm_kde(zn, z, h, kernel):
    """Return Kernel Density Estimation (KDE) non-conformity measure based on
    a certain kernel.

    Keyword arguments:
        zn the example on which to calculate the measure
        z: numpy.array containing examples one per row, excluding zn
        h: bandwidth
        K: kernel function, which can be in ['gaussian',] or can be a function
            calculating the kernel over a vector and returning a scalar
    """
    z_tmp = np.vstack((z, zn))
    (N, d) = z_tmp.shape
    if not isinstance(kernel, types.FunctionType):
        if kernel == 'gaussian':
            # Gaussian Kernel
            kernel = lambda u: np.exp(-0.5 * np.dot(u,u)) / np.sqrt(2 * np.pi)
        else:
            raise Exception('Non recognized Kernel function')
    r = - sum((kernel((zn - zi) / h) for zi in z_tmp)) / (N * (h ** d))

    return r
