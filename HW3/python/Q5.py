import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legroots, legder, legval
import os

# ==========================================
# 1. 基础工具函数 (网格与基函数)
# ==========================================

def legendre_gauss_lobatto(P):
    """ 计算 P 阶 Legendre-Gauss-Lobatto 节点和权重 """
    if P == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    c = np.zeros(P + 1)
    c[P] = 1
    cd = legder(c)
    inner_nodes = legroots(cd)
    nodes = np.hstack(([-1.0], inner_nodes, [1.0]))
    nodes = np.sort(nodes)
    LP_vals = legval(nodes, c)
    weights = 2 / (P * (P + 1) * LP_vals**2)
    return nodes, weights

# ==========================================
# 2. 初始条件与精确解定义
# ==========================================

def exact_solution_b(x, t):
    """ (b) 题的精确解 (也是初值) """
    # u(x,t) = [1 + exp(x/sqrt(6) - 5t/6)]^-2
    arg = x / np.sqrt(6) - 5 * t / 6
    val = np.zeros_like(x)
    # 防止溢出处理
    mask_0 = arg > 50
    mask_1 = arg < -50
    mask_mid = ~(mask_0 | mask_1)
    
    val[mask_0] = 0.0
    val[mask_1] = 1.0
    val[mask_mid] = (1 + np.exp(arg[mask_mid]))**(-2)
    return val

def ic_case_c1(x):
    """ (c) 题情形 1: Tanh 台阶 """
    return 0.5 * (1 - np.tanh(x))

def ic_case_c2(x):
    """ (c) 题情形 2: 复杂波包 """
    term1 = (1 + np.exp((x + 5) / 2))**2
    term2 = (1 + np.exp(x / 3))**2
    term3 = (1 + np.exp((x - 5) / 4))**2
    # 避免除以极大的数 (即结果为0)
    # 虽然 numpy float 处理除零通常没问题，但为了稳健
    with np.errstate(over='ignore'):
        denom = term1 * term2 * term3
        val = 1.0 / denom
    # 如果 denom 溢出为 inf， val 自动为 0，符合物理
    return val

# ==========================================
# 3. 通用求解器
# ==========================================

def run_fisher_solver(initial_func, exact_func=None, task_name="test", T_max=6.0, plot_times=None):
    """
    :param initial_func: t=0 时的初值函数 u0(x)
    :param exact_func: (可选) 用于计算误差的精确解函数 u(x,t)
    :param task_name: 任务名称，用于保存文件名
    :param T_max: 模拟总时长
    :param plot_times: 需要绘图的时间点列表
    """
    
    # --- 参数设置 ---
    N = 128
    L = 100.0
    tau = 1e-3
    steps = int(round(T_max / tau))
    
    if plot_times is None:
        plot_times = [1, 2, 3, 4, 5, 6]

    # --- 网格生成 ---
    xi, w_gl = legendre_gauss_lobatto(N)
    x = xi * L
    
    # --- 矩阵预计算 ---
    # 构造 P_mat: phi_k = L_k - L_{k+2}
    Leg = np.zeros((N + 3, N + 1)) 
    Leg[0, :] = 1.0
    Leg[1, :] = xi
    for k in range(1, N + 2):
        Leg[k+1, :] = ((2*k+1)*xi*Leg[k, :] - k*Leg[k-1, :]) / (k+1)
    L_vals = Leg.T 
    P_mat = L_vals[:, 0:N-1] - L_vals[:, 2:N+1]
    
    # 质量矩阵与刚度矩阵
    W_diag = np.diag(w_gl)
    M = P_mat.T @ W_diag @ P_mat
    k_idx = np.arange(N-1)
    S_diag = 4 * k_idx + 6
    
    # 线性求解器准备
    lam = tau / (2 * L**2)
    Mat_LHS = M + lam * np.diag(S_diag)
    Mat_RHS_Op = M - lam * np.diag(S_diag)
    
    inv_Mat_LHS = np.linalg.inv(Mat_LHS)
    inv_M = np.linalg.inv(M)
    Proj_Op = inv_M @ P_mat.T @ W_diag # 投影算子
    
    # --- 初始化 ---
    u = initial_func(x)
    u_bound = (1 - xi) / 2.0 # 边界辅助函数
    
    # 存储结果
    sols = {}
    errors = []
    
    # 如果需要画 t=0 (通常用于 (c) 题对比)
    if 0 in plot_times:
        sols[0] = u.copy()
    
    curr_t = 0.0
    print(f"--- 开始任务: {task_name} ---")
    
    # --- 时间步进 ---
    for step in range(1, steps + 1):
        # 1. 反应半步
        u = np.maximum(u, 0)
        dt_half = tau / 2.0
        exp_fac = np.exp(-dt_half)
        u = u / (u + (1 - u) * exp_fac)
        
        # 2. 扩散步
        v = u - u_bound
        v_hat_star = Proj_Op @ v
        rhs_cn = Mat_RHS_Op @ v_hat_star
        v_hat_next = inv_Mat_LHS @ rhs_cn
        v_next = P_mat @ v_hat_next
        u = v_next + u_bound
        
        # 3. 反应半步
        u = np.maximum(u, 0)
        u = u / (u + (1 - u) * exp_fac)
        
        curr_t += tau
        
        # --- 记录数据 ---
        if step % 100 == 0:
            t_int = int(round(curr_t))
            if abs(curr_t - t_int) < 1e-5 and t_int in plot_times:
                if t_int not in sols:
                    sols[t_int] = u.copy()
                    
                    # 如果有精确解，计算误差
                    if exact_func is not None:
                        u_ex = exact_func(x, curr_t)
                        err = np.max(np.abs(u - u_ex))
                        errors.append((t_int, err))
                        print(f"Time {t_int}: Error = {err:.4e}")

    # --- 绘图 (无标题) ---
    plt.figure(figsize=(10, 6))
    
    # 按时间排序
    sorted_times = sorted(sols.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_times)))
    
    for t, color in zip(sorted_times, colors):
        # 样式微调
        if t == 0:
            ls, lw, lbl = '--', 2.0, f't={t} (Initial)'
        else:
            ls, lw, lbl = '-', 1.5, f't={t}'
            
        plt.plot(x, sols[t], label=lbl, color=color, linestyle=ls, linewidth=lw)
    
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.xlim([-20, 40])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    save_dir = os.path.join(os.getcwd(), 'latex', "figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f"fisher_result_{task_name}.png")
    plt.savefig(filename, dpi=300)
    print(f"图像已保存: {filename}\n")
    # plt.show() 

# ==========================================
# 4. 主程序执行
# ==========================================

if __name__ == "__main__":
    
    # ------------------------------------
    # (b) 题: 验证精确解误差
    # ------------------------------------
    # 初值即为 exact_solution_b(x, 0)
    # 我们用 lambda 包装一下只传 x
    ic_b = lambda x: exact_solution_b(x, 0)
    
    run_fisher_solver(
        initial_func=ic_b,
        exact_func=exact_solution_b,
        task_name="b",
        T_max=6.0,
        plot_times=[1, 2, 3, 4, 5, 6]
    )
    
    # ------------------------------------
    # (c) 题 情形 1: Tanh 初值
    # ------------------------------------
    run_fisher_solver(
        initial_func=ic_case_c1,
        exact_func=None, # 无精确解，不算误差
        task_name="c1",
        T_max=10.0,      # 观察久一点
        plot_times=[0, 2, 4, 6, 8, 10]
    )

    # ------------------------------------
    # (c) 题 情形 2: 复杂初值
    # ------------------------------------
    run_fisher_solver(
        initial_func=ic_case_c2,
        exact_func=None,
        task_name="c2",
        T_max=10.0,
        plot_times=[0, 2, 4, 6, 8, 10]
    )