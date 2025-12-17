import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre
import os 

def LegendreP(n, x):
    '''计算n阶勒让德多项式在x处的值（递归法）'''
    if n == 0: return np.ones_like(x)
    elif n == 1: return x
    else:
        P_nm2, P_nm1 = np.ones_like(x), x
        for k in range(2, n + 1):
            P_n = ((2 * k - 1) * x * P_nm1 - (k - 1) * P_nm2) / k
            P_nm2, P_nm1 = P_nm1, P_n
        return P_n

def cal_LG(N):
    '''计算Legendre-Gauss节点和权重 (N+1个点)'''
    # Golub-Welsch算法: 构造对称三对角矩阵，特征值为节点
    # 对应于 P_{N+1} 的零点，矩阵大小 (N+1)x(N+1)
    b = np.array([np.sqrt(i**2 / (4*i**2 - 1)) for i in range(1, N+1)])
    A = np.diag(b, 1) + np.diag(b, -1)
    nodes, eigenvectors = np.linalg.eigh(A)
    weights = 2 * (eigenvectors[0, :] ** 2)
    return nodes, weights

def cal_LGL(N):
    '''计算Legendre-Gauss-Lobatto节点和权重 (N+1个点)'''
    if N < 2: raise ValueError("N must be at least 2")
    # 内部节点为 P'_N 的零点 (即Jacobi P_{N-1}^{(1,1)}的零点)
    # 矩阵大小 (N-1)x(N-1)，recurrence系数 b 长度 N-2
    b = np.array([np.sqrt(i*(i+2) / ((2*i+1)*(2*i+3))) for i in range(1, N-1)])
    A = np.diag(b, 1) + np.diag(b, -1)
    eigvals = np.linalg.eigvalsh(A)
    nodes = np.concatenate(([-1], eigvals, [1])) # 加上端点
    # 权重公式: w_j = 2 / (N(N+1) [P_N(x_j)]^2)
    weights = 2 / (N*(N+1) * (LegendreP(N, nodes) ** 2))
    return nodes, weights

if __name__ == "__main__":
    # --- 验证正确性 (N=50) ---
    N_check = 50
    
    # 1. 验证 LG
    my_lg_nodes, my_lg_weights = cal_LG(N_check)
    ref_lg_nodes, ref_lg_weights = np.polynomial.legendre.leggauss(N_check + 1)
    # 排序以对其
    idx = np.argsort(my_lg_nodes)
    err_lg_n = np.max(np.abs(my_lg_nodes[idx] - ref_lg_nodes))
    err_lg_w = np.max(np.abs(my_lg_weights[idx] - ref_lg_weights))

    # 2. 验证 LGL
    my_lgl_nodes, my_lgl_weights = cal_LGL(N_check)
    # 构造参考解: P'_N 的根 + 端点
    P_N = Legendre([0]*N_check + [1])
    ref_lgl_nodes = np.sort(np.concatenate(([-1, 1], P_N.deriv().roots())))
    ref_lgl_weights = 2 / (N_check * (N_check + 1) * P_N(ref_lgl_nodes)**2)
    
    idx = np.argsort(my_lgl_nodes)
    err_lgl_n = np.max(np.abs(my_lgl_nodes[idx] - ref_lgl_nodes))
    err_lgl_w = np.max(np.abs(my_lgl_weights[idx] - ref_lgl_weights))

    print(f"LG Error (N={N_check}): Node={err_lg_n:.2e}, Weight={err_lg_w:.2e}")
    print(f"LGL Error (N={N_check}): Node={err_lgl_n:.2e}, Weight={err_lgl_w:.2e}")

    # --- 可视化 (N=15) ---
    N_plot = 15
    lg_n, _ = cal_LG(N_plot)
    lgl_n, _ = cal_LGL(N_plot)
    
    save_dir = os.path.join(os.getcwd(), 'latex', 'figures')
    plt.figure(figsize=(10, 2.5))
    plt.plot(lg_n, np.zeros_like(lg_n), 'bx', markersize=8, markeredgewidth=1.5, label='Legendre-Gauss (LG)')
    plt.plot(lgl_n, np.zeros_like(lgl_n), 'r.', markersize=8, label='Legendre-Gauss-Lobatto (LGL)')
    plt.title(f'Node Distribution (N={N_plot})')
    plt.yticks([])
    plt.xlabel('x')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Q2_LG_LGL_nodes.png'), dpi=300)
    plt.show()