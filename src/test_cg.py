from numpy import *
import cg

def test_cg(n,etol):
    """
    Function to test the cg.solve_cg function.
    Inputs:
    n -- size of the test matrix
    etol -- tolerance required in the solution
    Outputs:
    None
    """
    A = diag(2*ones(n)) - diag(ones(n-1),1) - diag(ones(n-1),-1)
    x = arange(n)
    
    b = dot(A,x)
    x0 = 0.0*b

    xs, rvals = cg.solve_cg(A,b,x0,rtol=1.0e-5,maxits=10000)

    assert(max(abs(x-xs))<etol)
    

if __name__ == "__main__":
    test_cg(10,1.0e-4)
    test_cg(200,1.0e-6)
