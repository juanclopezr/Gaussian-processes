# ----------------------------------------------------------------------
#  MAIN PROGRAM - generated by the Rappture Builder
# ----------------------------------------------------------------------
import Rappture
import sys
import numpy as np
import math
import scipy.linalg
from scipy.stats import norm

# open the XML file containing the run parameters
io = Rappture.library(sys.argv[1])

#########################################################
# Get input values from Rappture
#########################################################

# get input value for input.number(lengthscale)
lengthscale = float(io.get('input.phase(values).group(inputpara).number(lengthscale).current'))

# get input value for input.number(signal_strength)
signal_strength = float(io.get('input.phase(values).group(inputpara).number(signal_strength).current'))

# get input value for input.number(noise_variance)
noise_var = float(io.get('input.phase(values).group(inputpara).number(noise_var).current'))

# get input value for input.number(n_samples)
n_samples = int(io.get('input.phase(values).integer(n_samples).current'))

# get input value for input.phase(values).boolean(manual)
# returns value as string "yes" or "no"
manual = io.get('input.phase(values).boolean(manual).current') == 'yes'

# get input value for input.phase(values).boolean(3dim)
# returns value as string "yes" or "no"
dim3 = io.get('input.phase(values).boolean(dim3).current') == 'yes'

# get input value for input.phase(values).group(axes).number(xminimum)
xminimum = float(io.get('input.phase(values).group(axes).number(xminimum).current'))
"""
# get input value for input.phase(values).group(axes).number(yminimum)
yminimum = float(io.get('input.phase(values).group(axes).number(yminimum).current'))
"""
# get input value for input.phase(values).group(axes).number(xmaximum)
xmaximum = float(io.get('input.phase(values).group(axes).number(xmaximum).current'))
"""
# get input value for input.phase(values).group(axes).number(ymaximum)
ymaximum = float(io.get('input.phase(values).group(axes).number(ymaximum).current'))
"""

#########################################################
#  Add your code here for the main body of your program
#########################################################

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
                     
def sample_gp(X, cov_fun=se_cov, num_samples=1, noise_variance=1e-12, **cov_params):
    """
    Sample a zero-mean Gaussian process at the inputs specified by X.
    
    :param X:              The input points on which we will sample the GP.
    :param cov_fun:        The covariance function we use.
    :param n_samples:    The number of samples to take.
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
    
x = np.zeros(500)
f = np.zeros((n_samples, 500))
y = np.zeros(50)
F = np.zeros((n_samples, 50, 50))

if dim3:
	io.put('output.mesh(thegrid).dim', '%d' % (2), append=1)
	io.put('output.mesh(thegrid).units', 'um', append=1)
	io.put('output.mesh(thegrid).grid.xaxis.min', '%d' % (xminimum), append=1)
	io.put('output.mesh(thegrid).grid.xaxis.max', '%d' % (xmaximum), append=1)
	io.put('output.mesh(thegrid).grid.xaxis.numpoints', '%d' % (50), append=1)
	io.put('output.mesh(thegrid).grid.yaxis.min', '%d' % (xminimum), append=1)
	io.put('output.mesh(thegrid).grid.yaxis.max', '%d' % (xmaximum), append=1)
	io.put('output.mesh(thegrid).grid.yaxis.numpoints', '%d' % (50), append=1)
	cov_fun=se_cov
	y = np.linspace(xminimum, xmaximum, 50)[:, None]
#y = np.linspace(xminimum, xmaximum, 50)[:, None]
	X1, X2 = np.meshgrid(y, y)
	X = np.hstack([X1.flatten()[:, None], X2.flatten()[:, None]])
	f1 = sample_gp(X, num_samples=n_samples, noise_variance=noise_var, cov_fun=cov_fun, s=signal_strength, ell=lengthscale)
	F = f1.reshape((n_samples, ) + X1.shape)
	
else:
	cov_fun=se_cov
	x = np.linspace(xminimum, xmaximum, 500)[:, None]
	f = sample_gp(x, num_samples=n_samples, noise_variance=noise_var, cov_fun=cov_fun, s=signal_strength, ell=lengthscale)

# spit out progress messages as you go along...
Rappture.Utils.progress(0, "Starting...")
Rappture.Utils.progress(5, "Loading data...")
Rappture.Utils.progress(50, "Half-way there")
Rappture.Utils.progress(100, "Done")

#########################################################
# Save output values back to Rappture
#########################################################

# save output value for output.curve(valuei), depending on the number of samples
# this shows all the points of the functions
if dim3:
	for j in xrange(n_samples):
		line = ""
		for i in xrange(y.shape[0]):
			for l in xrange(y.shape[0]):
				line += "%g " % F[j, i, l]
			line += "\n"
		temp = 'output.field(values%d)' % (j)
		temp1 = 'Function%d' % (j)
		io.put(temp+'.about.label', temp1, append=1)
		io.put(temp+'.component.mesh', 'output.mesh(thegrid)', append=1)
		io.put(temp+'.component.values', line, append=1)
else:
	for j in xrange(n_samples):
		line = ""
		for i in xrange(x.shape[0]):
			line += "%g %g\n" % (x[i],f.T[i, j])
		temp = 'output.curve(value%d)' % (j)
		temp1 = 'Function%d' % (j)
		io.put(temp+'.about.label', temp1, append=1)
		io.put(temp+'.about.description', 'Function from the Gaussian process', append=1)
		io.put(temp+'.xaxis.label', 'x', append=1)
		io.put(temp+'.xaxis.description', 'The domain of the function from the Gaussian process', append=1)
		io.put(temp+'.yaxis.label', 'f', append=1)
		io.put(temp+'.yaxis.description', 'The range of the function from the Gaussian process', append=1)
		io.put(temp+'.component.xy', line, append=1)

Rappture.result(io)
sys.exit()
