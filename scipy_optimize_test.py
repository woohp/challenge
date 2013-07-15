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
import math
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

def objective(vec, *args):
    x_size = N*D
    w_size = M*D
    t_size = M

    x = vec[:x_size]
    w = vec[x_size:x_size+w_size]
    t = vec[x_size+w_size:]

    x_section = numpy.sum(numpy.log(numpy.array([prob_x(x_i, 0) + prob_x(x_i, 1) for x_i in x])))
    w_section = numpy.sum(numpy.log(numpy.array(map(prob_w, w))))
    t_section = numpy.sum(numpy.log(numpy.array(map(prob_t, t))))

    l_section = 0
    for i in range(N):
        for j in range(M):
            # Note: A is the annotator image response matrix (N x M)
            if A[i][j] is not None:
                w_dot_x = numpy.dot(w[j*D:(j+1)*D], x[i*D:(i+1)*D])
                cdf_val = norm.cdf(w_dot_x-t[j])
                l_section += A[i][j] * math.log(cdf_val) + \
                             (1-A[i][j]) * math.log(1-cdf_val)
    return x_section + w_section + t_section + l_section

def prob_x(x, z_val):
    if z_val == 0:
        mu = -1
        theta = theta_0
        prob_z = 1 - beta
    else:
        mu = 1
        theta = theta_1
        prob_z = beta
    theta_sq = theta ** 2
    return (1 / (theta*math.sqrt(2*math.pi))) * math.exp(-(x-mu) / (2*theta_sq)) * prob_z

def prob_w(w):
    alpha_sq = alpha ** 2
    return (1 / (alpha*math.sqrt(2*math.pi))) * math.exp(-(w-1) / (2*alpha_sq))

def prob_t(t):
    gamma_sq = gamma ** 2
    return (1 / (gamma*math.sqrt(2*math.pi))) * math.exp(-t / (2*gamma_sq))
