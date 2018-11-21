from numpy import *

def solve_cg(A,b,x0,rtol,maxits):
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

    return x, rvals
