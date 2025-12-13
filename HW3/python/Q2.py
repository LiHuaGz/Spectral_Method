import numpy as np
from numpy.polynomial.legendre import Legendre

def LegendreP(n, x):
    '''
    Compute the value of the n-th order Legendre polynomial at x using recurrence relation.
    Parameters:
        n (int): Order of the Legendre polynomial.
        x (float or np.ndarray): Point(s) at which to evaluate the polynomial.
    Returns:
        P_n (float or np.ndarray): Value(s) of the n-th order Legendre polynomial at x.
    '''
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        P_nm2 = np.ones_like(x)  # P_0(x)
        P_nm1 = x               # P_1(x)
        for k in range(2, n + 1):
            P_n = ((2 * k - 1) * x * P_nm1 - (k - 1) * P_nm2) / k
            P_nm2, P_nm1 = P_nm1, P_n
        return P_n

def cal_LG(N):
    '''
    Calculate the nodes and weights for the Legendre-Gauss quadrature.
    Parameters:
        N (int): N+1 is the number of intetior nodes, thus A.shape = (N+1,N+1)
    Returns:
        nodes (np.ndarray): The Legendre-Gauss nodes.
        weights (np.ndarray): The Legendre-Gauss weights.
    '''
    b = np.array([np.sqrt(i**2 / (4*i**2 - 1)) for i in range(1, N+1)])
    A = np.diag(b, 1) + np.diag(b, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(A) # eigenvectors是单位化的
    nodes = eigenvalues
    weights = 2 * (eigenvectors[0, :] ** 2)
    return nodes, weights

def cal_LGL(N):
    '''
    Calculate the nodes and weights for the Legendre-Gauss-Lobatto quadrature.
    Parameters:
        N (int): N-1 is the number of interior nodes, thus A.shape = (N-1,N-1)
    Returns:
        nodes (np.ndarray): The Legendre-Gauss-Lobatto nodes.
        weights (np.ndarray): The Legendre-Gauss-Lobatto weights.
    '''
    if N < 2:
        raise ValueError("N must be at least 2 for Legendre-Gauss-Lobatto quadrature.")
    
    b = np.array([np.sqrt(i*(i+2) / ((2*i+1)*(2*i+3))) for i in range(1, N-1)]) # len=N-2
    A = np.diag(b, 1) + np.diag(b, -1)
    #eigenvalues = np.linalg.eigh(A)[0]
    eigenvalues = np.linalg.eigvalsh(A) # 只需要特征值，使用更高效的函数
    nodes = np.concatenate(([-1], eigenvalues, [1]))
    weights = 2 / (N*(N+1) * (LegendreP(N, nodes) ** 2))
    return nodes, weights

if __name__ == "__main__":
    N=1000
    nodes_LG, weights_LG = cal_LG(N)
    nodes_LG_real, weights_LG_real = np.polynomial.legendre.leggauss(N+1)
    print('Nodes LG max error:', np.max(np.abs(nodes_LG - nodes_LG_real)))
    print('Weights LG max error:', np.max(np.abs(weights_LG - weights_LG_real)))
    nodes_LGL, weights_LGL = cal_LGL(N)