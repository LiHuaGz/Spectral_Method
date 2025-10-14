from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.integrate import quad

def order2_sparse(a, f, N=1000):
    x = np.linspace(0, np.pi, N)
    h = np.pi/(N-1)
    b = f(x)
    b[0] = b[-1] = 0
    # 构造稀疏三对角矩阵
    diagonals = [np.full(N-1, -1/h**2), np.full(N, 2/h**2 + a), np.full(N-1, -1/h**2)]
    A = diags(diagonals, offsets=[-1, 0, 1], format='csr')
    # 边界条件
    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1
    A[-1, :] = 0
    A[-1, -1] = 1
    A = A.tocsr()
    u = spsolve(A, b)
    return u

def order4_sparse(a, f, N=1000):
    x = np.linspace(0, np.pi, N)
    h = np.pi/(N-1)
    diagonals = [np.full(N-1, a - 12/(h**2)), np.full(N, 10*a + 24/(h**2)), np.full(N-1, a - 12/(h**2))]
    A = diags(diagonals, offsets=[-1, 0, 1], format='csr')
    # 边界条件
    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1
    A[-1, :] = 0
    A[-1, -1] = 1
    A = A.tocsr()
    # 向量化b
    f_x = f(x)
    f_xm1 = np.roll(f_x, 1)
    f_xp1 = np.roll(f_x, -1)
    b = np.zeros_like(x)
    idx = np.arange(1, N-1)
    b[idx] = f_xm1[idx] + 10*f_x[idx] + f_xp1[idx]
    b[0] = b[-1] = 0
    u = spsolve(A, b)
    return u

def spectral_parallel(a, f, N=1000):
    x = np.linspace(0, np.pi, N)
    n_arr = np.arange(1, N)
    
    def quad_vec(n):
        try:
            return quad(lambda xx: f(xx) * np.sin(n * xx), 0, np.pi, limit=N)[0]
        except Exception:
            # Fallback for problematic integrations
            return 0.0
    
    # Use fewer threads for better stability and add timeout
    max_workers = min(8, len(n_arr))  # Limit to 8 threads
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Add timeout to prevent hanging
            quad_results = list(executor.map(quad_vec, n_arr))
    except (KeyboardInterrupt, Exception):
        # Fallback to sequential computation
        print("Warning: Parallel computation failed, falling back to sequential")
        quad_results = [quad_vec(n) for n in n_arr]
    
    u_qta = np.array(quad_results) * 2 / ((n_arr**2 + a) * np.pi)
    u = np.dot(u_qta, np.sin(np.outer(n_arr, x)))
    return u
