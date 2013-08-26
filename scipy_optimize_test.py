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

x_size = N*D
w_size = M*D

# Hyperparameters
theta_0 = 3
theta_1 = 3
beta = 0.5
gamma = 3
alpha = 3


# returns a single number, the objective function evaluated at vec
def objective(vec):
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
        cdf_val = norm.cdf(numpy.dot(w_j, x_i)-t[j])

        if l == 1:
            l_section += numpy.log(cdf_val)
        else:
            l_section += numpy.log(1-cdf_val)


    return x_section + w_section + t_section + l_section


# returns the gradient array
def gradient(vec):
    grad = numpy.empty(vec.shape, vec.dtype)
    x = vec[:x_size]
    w = vec[x_size:x_size+w_size]
    t = vec[x_size+w_size:]
    grad_x = grad[:x_size]
    grad_w = grad[x_size:x_size+w_size]
    grad_t = grad[x_size+w_size:]

    prob_x_0 = norm.pdf(x, loc=-1, scale=theta_0) * (1 - beta)
    prob_x_1 = norm.pdf(x, loc=1, scale=theta_1) * beta
    d_prob_x_0 = gaussian_derivative(x, loc=-1, scale=theta_0) * (1 - beta)
    d_prob_x_1 = gaussian_derivative(x, loc=1, scale=theta_1) * beta
    grad_x = (d_prob_x_0 + d_prob_x_1) / (prob_x_0 + prob_x_1)
    grad_w = gaussian_derivative(w, loc=1, scale=alpha) / norm.pdf(w, loc=1, scale=alpha)
    grad_t = gaussian_derivative(t, loc=0, scale=gamma) / norm.pdf(t, loc=0, scale=gamma)

    for i, j, l in A:
        x_i = x[i*D:(i+1)*D]
        w_j = w[j*D:(j+1)*D]
        grad_x_i = grad_x[i*D:(i+1)*D]
        grad_w_j = grad_w[j*D:(j+1)*D]
        grad_t_j = grad_t[j]

        arg = numpy.dot(w_j, x_i)-t[j]
        pdf_val = norm.pdf(arg)
        cdf_val = norm.cdf(arg)

        if l == 1:
            grad_ = pdf_val / cdf_val
        else:
            grad_ = -pdf_val / (1 - cdf_val)

        grad_x_i += grad_ * numpy.sum(w_j)
        grad_w_j += grad_ * numpy.sum(x_i)
        grad_t_j += grad_

    return grad

def gaussian_derivative(x, loc=0, scale=1):
    return norm.pdf(x, loc=loc, scale=scale) * (loc - x) / (scale * scale)


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
    print gradient(numpy.zeros(N*D + M*D + M))
