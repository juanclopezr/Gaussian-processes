
# coding: utf-8

# In[1]:

import numpy as np
import math
import scipy.linalg
from scipy.stats import norm


# In[2]:

def se_cov(x1, x2, s=1., ell=1.):
    """
    A function the computes the SE covariance:
    
    :param x1:        One input point (1D array).
    :param x2:        Another input point (1D array).
    :param s:         The signal strength (> 0).
    :param ell:       The length scale(s). Either a positive number or a 1D array of
                      positive numbers.
    :returns:         The value of the SE covariance evaluated at x1 and x2 with
                      signal strength s and length scale(s) ell.
    
    .. pre::
        
        + x1 must have the same dimensions as x2
        + ell must either be a float or a 1D array with the same dimensions as x1
    """
    tmp = (x1 - x2) / ell
    return s ** 2 * math.exp(-0.5 * np.dot(tmp, tmp))


# In[3]:

def cov_mat(X1, X2, cov_fun=se_cov, **cov_params):
    """
    Compute the cross covariance matrix of `X1` and `X2` for the covariance
    function `cov_fun`.
    
    :param X1:           A matrix of input points (n1 x d)
    :param X2:           A matrix of input points (n2 x d)
    :param cov_fun:      The covariance function to use
    :param cov_param:    Any parameters that we should pass to the covariance
                         function `cov_fun`.
    """
    X1 = np.array(X1)
    X2 = np.array(X2)
    return np.array([[cov_fun(X1[i, :], X2[j, :], **cov_params) for j in xrange(X2.shape[0])]
                     for i in xrange(X1.shape[0])])


# In[4]:

def sample_gp(X, cov_fun=se_cov, num_samples=1, noise_variance=1e-12, **cov_params):
    """
    Sample a zero-mean Gaussian process at the inputs specified by X.
    
    :param X:              The input points on which we will sample the GP.
    :param cov_fun:        The covariance function we use.
    :param num_samples:    The number of samples to take.
    :param noise_variance: The noise of the process.
    :param cov_params:     Any parameters of the covariance function.
    """
    # Compute the covariance matrix:
    K = cov_mat(X, X, cov_fun=cov_fun, **cov_params) + noise_variance * np.eye(X.shape[0])
    # Compute the cholesky of this:
    L = scipy.linalg.cholesky(K, lower=True) 
    # Take a sample of standard normals
    z = norm.rvs(size=(X.shape[0], num_samples))
    # Build the sample from the multivariate normal
    f = np.dot(L, z)
    return f.T


# In[16]:

cov_fun=se_cov
s = 1.
ell = 1.
x = np.linspace(-3, 3, 50)[:, None]
f = sample_gp(x, num_samples=5, cov_fun=cov_fun, s=s, ell=ell)
for j in xrange(5):
    for i in xrange(x.shape[0]):
        print "%g %g" % (x[i],f.T[i, j])


# In[ ]:



