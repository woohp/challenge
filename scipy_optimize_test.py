'''
fmin_cg:
    args: f = objective function
          x = initial values of parameters to optimize
          fprime = derivative function (returns list that is derivative w.r.t each parameter)
          # Debug flags
          full_output = Boolean to determine whether or not to return debug vars
          callback = Function called after every iteration
          maxiter = Maximum number of iters to run (Default: 200 * len(x))
    return: xopt = Optimal values for x
            fopt = Optimal objective value
            func_calls = Number of function calls made
            grad_calls = Number of gradient calls made
            warnflag = Warnings (0 = success, 1 = max iter reached, 2 = gradient and/or function calls were not changing)
'''
import numpy
from scipy.stats import norm

# Dimensions
N = 10 # number of annotators. Indexed by j
M = 10 # number of questions. Indexed by i
D = 1 # dimension of x vector 

# Hyperparameters
theta_0 = 3
theta_1 = 3
beta = 0.5
gamma = 3
alpha = 3


def objective(vec):
    x_size = N*D
    w_size = M*D

    x = vec[:x_size]
    w = vec[x_size:x_size+w_size]
    t = vec[x_size+w_size:]

    prob_x_0 = norm.pdf(x, loc=-1, scale=theta_0) * (1 - beta)
    prob_x_1 = norm.pdf(x, loc=1, scale=theta_1) * beta
    x_section = numpy.sum(numpy.log(prob_x_0 + prob_x_1))
    w_section = numpy.sum(numpy.log(norm.pdf(w, loc=1, scale=alpha)))
    t_section = numpy.sum(numpy.log(norm.pdf(t, loc=0, scale=gamma)))

    l_section = 0
    # Note: A is matrix of size (L, 3) where L is the number of labels. It is the array of (i, j, l)
    for i, j, l in A:
        x_i = x[i*D:(i+1)*D]
        w_j = w[j*D:(j+1)*D]
        w_dot_x = numpy.dot(w_j, x_i)
        cdf_val = norm.cdf(w_dot_x-t[j])
        if l == 1:
            l_section += numpy.log(cdf_val)
        else:
            l_section += numpy.log(1-cdf_val)


    return x_section + w_section + t_section + l_section


if __name__ == '__main__':
    A = numpy.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1] ])
    A = numpy.array([(i, j, A[i, j]) for i in xrange(A.shape[0]) for j in xrange(A.shape[1])])
    print objective(numpy.zeros(N*D + M*D + M))
