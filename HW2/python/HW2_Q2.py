import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, LinearOperator
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import multiprocessing

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun']  # 微软雅黑、宋体

def FDM(p, q, f, a, b, N):
    '''
    p,q,f: functions of x
    a,b: interval endpoints
    N: number of steps
    '''
    x = np.linspace(a, b, N, endpoint=False)  # 不包括b点
    h = (b - a) / N  # 因为只有N个点
    A = np.zeros((N, N))    # 由于u_0 = u_N,所以矩阵A是N*N的
    B = np.zeros(N)
    
    # 向量化操作赋值A, B
    x_backhalf = x - h / 2
    x_forehalf = x + h / 2
    
    # 计算所有的 p 和 q 值
    p_back = np.array([p(xk) for xk in x_backhalf])
    p_fore = np.array([p(xk) for xk in x_forehalf])
    q_vals = np.array([q(xk) for xk in x])
    f_vals = np.array([f(xk) for xk in x])
    
    # 对角线元素 (正确的二阶差分格式)
    np.fill_diagonal(A, (p_back + p_fore) / h**2 + q_vals)
    
    # 下对角线元素 (周期边界条件)
    A[np.arange(1, N), np.arange(0, N-1)] = -p_back[1:] / h**2
    A[0, -1] = -p_back[0] / h**2  # u_-1 = u_N-1
    
    # 上对角线元素 (周期边界条件)
    A[np.arange(0, N-1), np.arange(1, N)] = -p_fore[:-1] / h**2
    A[-1, 0] = -p_fore[-1] / h**2  # u_N = u_0
    
    # 右端项 (不需要乘以 h²)
    B = f_vals
    
    u = scipy.linalg.solve(A, B)
    return u

def F_spectral(p, q, f, a=0, b=2*np.pi, N=100, return_iter=False):
    '''
    p,q,f: functions of x
    a,b: interval endpoints
    N: number of modes
    return_iter: if True, return (u, iteration_count) instead of just u
    Note: since the linear system is black-boxed, we use Conjugate-Gradient to solve it.
    '''
    # initialize the vector k needed for calculating derivatives
    k = np.zeros(N, dtype=np.complex128)
    k[:N//2], k[N//2], k[N//2+1:] = 1j * np.arange(0, N//2), 0, 1j * np.arange(-N//2+1, 0)

    # initialize P, Q and F vectors in spectral space
    x = np.linspace(a, b, N, endpoint=False)
    P = np.array([p(xk) for xk in x])
    Q = np.array([q(xk) for xk in x])
    F = np.array([f(xk) for xk in x])

    # calculate FFT
    F_hat = np.fft.fft(F)

    # def linear operator
    def L1_operator(P, u_hat):
        u_x = np.fft.ifft(k * u_hat)    # u' in physical space
        Pu_x_hat = np.fft.fft(P * u_x)     # Pu' in spectral space
        Pu_x_x_hat = k * Pu_x_hat    # (Pu')' in spectral space
        return - Pu_x_x_hat
    
    def L2_operator(Q, u_hat):
        u = np.fft.ifft(u_hat)
        Qu_hat = np.fft.fft(Q * u)
        return Qu_hat

    def L_operator(P, Q, u_hat):
        return L1_operator(P, u_hat) + L2_operator(Q, u_hat)
    
    # use Conjugate-Gradient method to solve the linear system
    # Create a LinearOperator for the CG method
    A = LinearOperator((N, N), matvec=lambda u_hat: L_operator(P, Q, u_hat), dtype=np.complex128)
    u_hat = np.zeros(N, dtype=np.complex128)
    
    # 用于记录迭代次数
    iteration_count = [0]
    def callback(xk):
        iteration_count[0] += 1

    u_hat, info = cg(A, F_hat, x0=u_hat, tol=1e-10, maxiter=N, callback=callback)
    u = np.fft.ifft(u_hat)

    if return_iter:
        return u, iteration_count[0]
    return u

def F_collocation(p, p_prime, q, f, a=0, b=2*np.pi, N=100, return_iter=False):
    '''
    p,p_prime,q,f: functions of x
    a,b: interval endpoints
    N: number of collocation points
    return_iter: if True, return (u, iteration_count) instead of just u
    '''
    # initialize differentiation matrices
    x = np.linspace(a, b, N, endpoint=False)  # 注意周期性, 不包括b点
    
    # 使用向量化操作构建微分矩阵
    i_indices = np.arange(N).reshape(-1, 1)  # 列向量
    j_indices = np.arange(N).reshape(1, -1)  # 行向量
    
    # 计算 i - j 和 (-1)^(i+j)
    diff = i_indices - j_indices
    sign = (-1) ** (i_indices + j_indices)
    
    # 非对角元素
    theta = np.pi * diff / N
    mask = (i_indices != j_indices)
    
    D1 = np.zeros((N, N))
    D2 = np.zeros((N, N))
    
    # 计算非对角元素
    with np.errstate(divide='ignore', invalid='ignore'):
        D1 = np.where(mask, sign / np.tan(theta) / 2, 0)
        D2 = np.where(mask, -sign / np.sin(theta)**2 / 2, 0)
    
    # 对角元素
    np.fill_diagonal(D1, 0)
    np.fill_diagonal(D2, -N**2 / 12 - 1/6)

    # initialize P, P', Q and f vectors
    P = np.diag([p(xk) for xk in x])
    P_prime = np.diag([p_prime(xk) for xk in x])
    Q = np.diag([q(xk) for xk in x])
    F = np.array([f(xk) for xk in x])
    
    # initialize system matrix A and right-hand side B
    A = -P_prime @ D1 - P @ D2 + Q
    B = F
    
    # 普通的解方程法
    if not return_iter:
        u = scipy.linalg.solve(A, B)

    # 共轭梯度法
    # 用于记录迭代次数
    if return_iter:
        iteration_count = [0]
        def callback(xk):
            iteration_count[0] += 1
        u, info = cg(A, B, tol=1e-10, maxiter=N, callback=callback)
        return u, iteration_count[0]
    return u

def compute_L2_error(u_numeric, u_exact, h):
    """计算离散L²误差, 注意要乘以sqrt(h)"""
    return np.linalg.norm(u_numeric - u_exact, 2) * np.sqrt(h)

def compute_H1_error(u_numeric, u_exact, h):
    """计算离散H¹误差, 注意要乘以sqrt(h)"""
    du_numeric = np.gradient(u_numeric) / h
    du_exact = np.gradient(u_exact) / h
    return np.linalg.norm(du_numeric - du_exact, 2) * np.sqrt(h)

if __name__ == "__main__":
    # Example usage
    u = lambda x: np.exp(np.sin(8*x))
    p = lambda x: 2+np.sin(x)
    p_prime = lambda x: np.cos(x)
    q = lambda x: np.sin(x)**2
    f = lambda x: np.exp(np.sin(8*x))*(-8*np.cos(x)*np.cos(8*x)+64*(2+np.sin(x))*(np.sin(8*x)-np.cos(8*x)**2)+np.sin(x)**2)
    a = 0
    b = 2*np.pi
    N_list = [2**i for i in range(6, 13)]
    h = [(b - a) / N for N in N_list]
    current_path = os.getcwd()
    savepath = os.path.join(current_path, 'latex/figures')
    os.makedirs(savepath, exist_ok=True)

    
    # 定义计算函数，用于并行化
    def compute_for_N(N):
        """对单个N值计算所有方法的结果"""
        x = np.linspace(a, b, N, endpoint=False)    # x0, x1, ..., x_{N-1}
        u_exact_val = u(x)
        u_FDM_val = FDM(p, q, f, a, b, N)
        u_spec, iter_spec = F_spectral(p, q, f, a, b, N, return_iter=True)
        u_coll_GE = F_collocation(p, p_prime, q, f, a, b, N, return_iter=False)
        u_coll_CG, iter_coll = F_collocation(p, p_prime, q, f, a, b, N, return_iter=True)
        
        return {
            'N': N,
            'u_exact': u_exact_val,
            'u_FDM': u_FDM_val,
            'u_F_spectral': u_spec,
            'spectral_iter': iter_spec,
            'u_F_collocation_GE': u_coll_GE,
            'u_F_collocation_CG': u_coll_CG,
            'collocation_iter_CG': iter_coll
        }
    
    # 使用多线程并行计算
    # 自动检测 CPU 核心数，设置最优线程数
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = min(cpu_count, len(N_list))  # 不超过任务数和核心数
    
    print(f"检测到 {cpu_count} 个 CPU 核心")
    print(f"使用 {optimal_workers} 个线程并行计算...")
    
    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        results = list(executor.map(compute_for_N, N_list))
    
    print("并行计算完成！")
    
    # 整理结果
    u_exact, u_FDM, u_F_spectral, u_F_collocation_GE, u_F_collocation_CG = [],[],[],[],[]
    spectral_iters, collocation_iters = [], []
    
    for result in results:
        u_exact.append(result['u_exact'])
        u_FDM.append(result['u_FDM'])
        u_F_spectral.append(result['u_F_spectral'])
        spectral_iters.append(result['spectral_iter'])
        u_F_collocation_GE.append(result['u_F_collocation_GE'])
        u_F_collocation_CG.append(result['u_F_collocation_CG'])
        collocation_iters.append(result['collocation_iter_CG'])

    # question 2 (a) error analysis and plotting
    # compute u_FDM L^2 and H^1 errors
    FDM_L2_errors = []
    FDM_H1_errors = []
    for i in range(len(N_list)):
        N = N_list[i]
        L2_error = compute_L2_error(u_FDM[i], u_exact[i], (b - a) / N)
        H1_error = compute_H1_error(u_FDM[i], u_exact[i], (b - a) / N)
        FDM_L2_errors.append(L2_error)
        FDM_H1_errors.append(H1_error)
    # 计算收敛阶(斜率) - 使用对数空间的线性拟合
    log_h = np.log(h)
    log_L2_errors = np.log(FDM_L2_errors)
    log_H1_errors = np.log(FDM_H1_errors)
    slope_L2 = np.polyfit(log_h, log_L2_errors, 1)[0]   # L^2 误差的斜率
    slope_H1 = np.polyfit(log_h, log_H1_errors, 1)[0]   # H^1 误差的斜率
    # plot error-h log-log errors, save as PNG
    plt.figure()
    plt.loglog(h, FDM_L2_errors, 'o-', linewidth=2, markersize=8, 
               markerfacecolor='white', markeredgewidth=1.5, 
               label=f'$L^2$ 误差 (斜率: {slope_L2:.2f})')
    plt.loglog(h, FDM_H1_errors, 's--', linewidth=2, markersize=8, 
               markerfacecolor='gray', markeredgewidth=1.5, 
               label=f'$H^1$ 误差 (斜率: {slope_H1:.2f})')
    plt.xlabel('网格尺寸 (h)')
    plt.ylabel('误差')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(savepath, 'Q2a_FDM_Error_Analysis.png'))
    plt.show()

    # question 2 (b) error analysis and plotting
    # compute u_F_spectral L^2 errors
    F_spectral_L2_errors = []
    for i in range(len(N_list)):
        N = N_list[i]
        L2_error = compute_L2_error(u_F_spectral[i], u_exact[i], (b - a) / N)
        F_spectral_L2_errors.append(L2_error)

    # plot error-N semilog errors, save as PNG
    plt.figure()
    plt.semilogy(N_list, F_spectral_L2_errors, 'o-')
    plt.xlabel('配点数 (N)')
    plt.ylabel('$L^2$ 误差')
    plt.grid()
    plt.savefig(os.path.join(savepath, 'Q2b_F_spectral_Error_Analysis.png'))
    plt.show()

    # question 2 (c) CG iterations vs N for reaching 10-digit precision ,F_spectral
    plt.figure()
    plt.plot(N_list, spectral_iters, 'o-')
    plt.xlabel('配点数 (N)')
    plt.ylabel('CG 迭代次数')
    plt.grid(True)
    plt.savefig(os.path.join(savepath, 'Q2c_F_spectral_CG_Iterations.png'))
    plt.show()

    # question 2 (d) error analysis and plotting
    # compute u_F_collocation L^2 errors, for GE method
    F_collocation_L2_errors = []
    for i in range(len(N_list)):
        N = N_list[i]
        L2_error = compute_L2_error(u_F_collocation_GE[i], u_exact[i], (b - a) / N)
        F_collocation_L2_errors.append(L2_error)
    # plot error-N semilog errors, save as PNG
    plt.figure()
    plt.semilogy(N_list, F_collocation_L2_errors, 'o-')
    plt.xlabel('配点数 (N)')
    plt.ylabel('$L^2$ 误差')
    plt.grid()
    plt.savefig(os.path.join(savepath, 'Q2d_F_collocation_Error_Analysis_GE.png'))
    plt.show()
    # compute u_F_collocation L^2 errors, for CG method
    F_collocation_L2_errors_CG = []
    for i in range(len(N_list)):
        N = N_list[i]
        L2_error = compute_L2_error(u_F_collocation_CG[i], u_exact[i], (b - a) / N)
        F_collocation_L2_errors_CG.append(L2_error)
    # plot error-N semilog errors, save as PNG
    plt.figure()
    plt.semilogy(N_list, F_collocation_L2_errors_CG, 'o-')
    plt.xlabel('配点数 (N)')
    plt.ylabel('$L^2$ 误差')
    plt.grid()
    plt.savefig(os.path.join(savepath, 'Q2d_F_collocation_Error_Analysis_CG.png'))
    plt.show()

    # question 2 (e) CG iterations vs N for reaching 10-digit precision, F_collocation
    plt.figure()
    plt.plot(N_list, collocation_iters, 'o-')
    plt.xlabel('配点数 (N)')
    plt.ylabel('CG 迭代次数')
    plt.grid(True)
    plt.savefig(os.path.join(savepath, 'Q2e_F_collocation_CG_Iterations.png'))
    plt.show()