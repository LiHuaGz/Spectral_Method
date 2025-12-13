import numpy as np
import scipy.special as sp
import scipy.linalg as la
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os

# ==========================================
# 第一部分：通用函数 (真解定义)
# ==========================================

def u_exact_func(x, gamma):
    """
    真解 u(x):
    x <= 0: 0
    x > 0:  x^gamma
    """
    val = np.zeros_like(x)
    mask = x > 0
    val[mask] = x[mask]**gamma
    return val

# ==========================================
# 第二部分：问题 4(a) - Legendre-Galerkin 方法
# ==========================================

def get_lgl_nodes_weights(N):
    """ 计算 Legendre-Gauss-Lobatto 节点和权重 """
    if N == 1: return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    inner_roots, _ = sp.roots_jacobi(N-1, 1, 1)
    nodes = np.zeros(N+1)
    nodes[0], nodes[-1] = -1.0, 1.0
    nodes[1:-1] = inner_roots
    LN = sp.eval_legendre(N, nodes)
    weights = 2 / (N * (N+1) * LN**2)
    return nodes, weights

def compute_legendre_matrices(N):
    k = np.arange(N-1)
    # 刚度矩阵 S
    diag_S = 4*k + 6
    S = np.diag(diag_S)
    # 质量矩阵 M
    diag_M = 2/(2*k+1) + 2/(2*k+5)
    M = np.diag(diag_M)
    off_diag = -2/(2*k[:-2]+5)
    for i in range(N-3):
        M[i, i+2] = off_diag[i]
        M[i+2, i] = off_diag[i]
    return M, S

def f_rhs_a(x, gamma):
    if x <= 0: return 0.0
    return x**gamma - gamma*(gamma-1)*x**(gamma-2)

def legendre_basis_val(x, k):
    vals = sp.eval_legendre([k, k+2], x)
    return vals[0] - vals[1]

def solve_4a(N, gamma):
    M, S = compute_legendre_matrices(N)
    F = np.zeros(N-1)
    
    # (u_b, phi_j) 的解析投影
    if N-1 > 0: F[0] -= 1.0
    if N-1 > 1: F[1] -= 1.0/3.0
    
    # (f, phi_j) 数值积分
    for j in range(N-1):
        integrand = lambda x: f_rhs_a(x, gamma) * legendre_basis_val(x, j)
        val, _ = quad(integrand, 0, 1, limit=100)
        F[j] += val
        
    coeffs = la.solve(M + S, F)
    
    # 计算误差
    nodes, weights = get_lgl_nodes_weights(N)
    P = np.zeros((N+1, len(nodes)))
    P[0] = 1.0; P[1] = nodes
    for k in range(1, N):
        P[k+1] = ((2*k+1)*nodes*P[k] - k*P[k-1]) / (k+1)
        
    u_N = (nodes + 1) / 2.0
    for k in range(N-1):
        phi_k = P[k] - P[k+2]
        u_N += coeffs[k] * phi_k
        
    u_ex = u_exact_func(nodes, gamma)
    error = np.sqrt(np.sum((u_ex - u_N)**2 * weights))
    return error

# ==========================================
# 第三部分：问题 4(b) - Chebyshev 配点法
# ==========================================

def cheb_nodes_weights(N):
    theta = np.pi * np.arange(N+1) / N
    x = np.cos(theta)
    w = np.full(N+1, np.pi/N)
    w[0] = np.pi/(2*N); w[-1] = np.pi/(2*N)
    return x, w

def cheb_diff_matrix(N):
    x, _ = cheb_nodes_weights(N)
    c = np.ones(N+1); c[0]=2; c[-1]=2
    c = c * (-1)**np.arange(N+1)
    X = np.tile(x, (N+1, 1))
    dX = X.T - X
    with np.errstate(divide='ignore', invalid='ignore'):
        D = (c[:, None] / c[None, :]) / (dX + np.eye(N+1))
    D = D - np.diag(D.sum(axis=1))
    return x, D

def f_rhs_b(x, gamma):
    f = np.zeros_like(x)
    mask = x > 0
    xm = x[mask]
    term1 = xm**(gamma + 2)
    term2 = gamma * np.exp(xm) * (xm**(gamma-2)) * (xm + gamma - 1)
    f[mask] = term1 - term2
    return f

def solve_4b(N, gamma, bc_type):
    x, D = cheb_diff_matrix(N)
    D2 = D @ D
    X2 = np.diag(x**2)
    E = np.diag(np.exp(x))
    A = X2 - E @ D - E @ D2
    b = f_rhs_b(x, gamma)
    
    # 边界条件
    A[0, :] = 0.0; A[0, 0] = 1.0; b[0] = 1.0
    
    if bc_type == 1: # u(-1)=0
        A[N, :] = 0.0; A[N, N] = 1.0; b[N] = 0.0
    elif bc_type == 2: # u(-1)-u'(-1)=0
        row = -D[N, :].copy()
        row[N] += 1.0
        A[N, :] = row; b[N] = 0.0
        
    u_N = la.solve(A, b)
    _, weights = cheb_nodes_weights(N)
    u_ex = u_exact_func(x, gamma)
    error = np.sqrt(np.sum((u_ex - u_N)**2 * weights))
    return error

# ==========================================
# 主程序
# ==========================================

def main():
    save_dir = os.path.join(os.getcwd(), 'latex', "figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gammas = [4, 5, 6]
    Ns_a = [2**i for i in range(4, 10)]
    Ns_b = [2**i for i in range(4, 10)] 
    
    results = {
        '4a': {g: [] for g in gammas},
        '4b_bc1': {g: [] for g in gammas},
        '4b_bc2': {g: [] for g in gammas}
    }
    
    print("Calculating 4(a)...")
    for g in gammas:
        for n in Ns_a:
            results['4a'][g].append(solve_4a(n, g))
            
    print("Calculating 4(b)...")
    for g in gammas:
        for n in Ns_b:
            results['4b_bc1'][g].append(solve_4b(n, g, bc_type=1))
            results['4b_bc2'][g].append(solve_4b(n, g, bc_type=2))

    # ==========================================
    # 定义绘图样式 (黑白打印优化)
    # ==========================================
    # 不同的 gamma 使用不同的 标记(marker) 和 线型(linestyle)
    styles = {
        4: {'marker': 'o', 'linestyle': '-',  'color': 'black', 'markerfacecolor': 'none', 'markeredgewidth': 1}, # 实线，空心圆
        5: {'marker': 's', 'linestyle': '--', 'color': 'black', 'markerfacecolor': 'none', 'markeredgewidth': 1}, # 虚线，空心方块
        6: {'marker': '^', 'linestyle': '-.', 'color': 'black', 'markerfacecolor': 'none', 'markeredgewidth': 1}  # 点划线，空心三角
    }

    # ================= 绘图 1: 4(a) =================
    plt.figure(figsize=(8, 6))
    for g in gammas:
        slope, _ = np.polyfit(np.log10(Ns_a), np.log10(results['4a'][g]), 1)
        plt.loglog(Ns_a, results['4a'][g], 
                   label=f'$\gamma={g}$, slope={slope:.2f}',
                   linewidth=1.5, markersize=8, **styles[g])
    
    plt.xlabel('N', fontsize=12)
    plt.ylabel(r'Error $\|u-u_N\|_{N,w}$', fontsize=12)
    plt.grid(True, which='major', linestyle='-', alpha=0.5, color='gray') # 主网格
    plt.grid(True, which='minor', linestyle=':', alpha=0.3, color='gray') # 次网格
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'result_4a.png'), dpi=300)
    print("Figure saved: result_4a.png")
    # plt.show() 

    # ================= 绘图 2: 4(b) =================
    plt.figure(figsize=(14, 6))
    
    # 子图 1: Dirichlet
    plt.subplot(1, 2, 1)
    for g in gammas:
        slope, _ = np.polyfit(np.log10(Ns_b[:-1]), np.log10(results['4b_bc1'][g][:-1]), 1)
        plt.loglog(Ns_b, results['4b_bc1'][g], 
                   label=f'$\gamma={g}$, slope={slope:.2f}',
                   linewidth=1.5, markersize=8, **styles[g])
    plt.title(r'4(b) Chebyshev: BC $u(-1)=0$', fontsize=13)
    plt.xlabel('N', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.grid(True, which='major', linestyle='-', alpha=0.5, color='gray')
    plt.grid(True, which='minor', linestyle=':', alpha=0.3, color='gray')
    plt.legend(fontsize=11)

    # 子图 2: Robin
    plt.subplot(1, 2, 2)
    for g in gammas:
        slope, _ = np.polyfit(np.log10(Ns_b[:-1]), np.log10(results['4b_bc2'][g][:-1]), 1)
        plt.loglog(Ns_b, results['4b_bc2'][g], 
                   label=f'$\gamma={g}$, slope={slope:.2f}',
                   linewidth=1.5, markersize=8, **styles[g])
    plt.title(r'4(b) Chebyshev: BC $u(-1)-u\'(-1)=0$', fontsize=13)
    plt.xlabel('N', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.grid(True, which='major', linestyle='-', alpha=0.5, color='gray')
    plt.grid(True, which='minor', linestyle=':', alpha=0.3, color='gray')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'result_4b.png'), dpi=300)
    print("Figure saved: result_4b.png")
    plt.show()

if __name__ == "__main__":
    main()