import numpy as np


def solve_cg(A, b, x, rtol, maxits):
    """
    Function to use the conjugate gradient algorithm to
    solve the equation Ax = b for symmetric positive definite A.
    Inputs:
    A -- An nxn matrix stored as a rank-2 numpy array
    b -- A length n vector stored as a rank-1 numpy array
    x0 -- The initial guess, length n vector stored as a rank-1 numpy array
    rtol -- a tolerance, algorithm should stop if l2-norm of 
    the residual r=Ax-b drops below this value.
    maxits -- Stop if the tolerance has not been reached after this number
    of iterations
    
    Outputs:
    x -- the approximate solution
    rvals -- a numpy array containing the l2 norms of the residuals
    r=Ax-b at each iteration
    """
    rvals = rtol + 1
    first_iteration = True
    r = b
    while rvals > rtol:
        if first_iteration:
            p = r.copy()
            first_iteration = False
        else:
            p = r + np.dot(r, r) / np.dot(r_old, r_old) * p
        alpha = np.dot(r, r) / np.dot(p, np.dot(A, p))
        x = x + alpha * p
        r_old = r.copy()
        r = r - alpha * np.dot(A, p)
        rvals = np.sqrt(np.dot(r, r))
    return x, rvals
