import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun']  # 微软雅黑、宋体

def solve_Burger(epsilon=0.03,t_end=1,Nt=256,a=0,b=2*np.pi,Nx=256,u0=None,save_history=False):
    '''
    使用RK4法和谱方法求解粘性Burgers方程
    u_t = epsilon*u_xx + u*u_x, x in [a,b], t in [0,t]
    '''
    if u0 is None:
        u0 = lambda x: np.exp(-10*np.cos(x/2)**2)
    # 初始化空间和时间离散
    x = np.linspace(a, b, Nx, endpoint=False)
    dx = (b - a) / Nx
    dt = t_end / Nt

    # 初始条件
    u = u0(x)
    
    # 如果需要保存历史数据，初始化数组
    if save_history:
        u_history = np.zeros((Nt+1, Nx))
        u_history[0, :] = u.real

    # 初始化计算u_x和u_xx所需要的向量k^1,k^2
    k_1, k_2 = np.zeros(Nx, dtype=np.complex128), np.zeros(Nx, dtype=np.complex128)
    k_1[:Nx//2], k_1[Nx//2], k_1[Nx//2+1:] = 1j * np.arange(0, Nx//2), 0, 1j * np.arange(-Nx//2+1, 0)
    k_2[:Nx//2], k_2[Nx//2], k_2[Nx//2+1:] = - (np.arange(0, Nx//2))**2, -(Nx//2)**2/2, - (np.arange(-Nx//2+1, 0))**2

    for n in range(Nt):
        u_hat = np.fft.fft(u)

        def rhs(u_hat):
            u = np.fft.ifft(u_hat)
            u_x = np.fft.ifft(k_1 * u_hat)
            u_xx = np.fft.ifft(k_2 * u_hat)
            return np.fft.fft(epsilon * u_xx + u * u_x)

        # Runge-Kutta 4阶时间步进
        k1 = rhs(u_hat)
        k2 = rhs(u_hat + 0.5 * dt * k1)
        k3 = rhs(u_hat + 0.5 * dt * k2)
        k4 = rhs(u_hat + dt * k3)

        u_hat += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        u = np.fft.ifft(u_hat)
        
        if save_history:
            u_history[n+1, :] = u.real
    
    if save_history:
        return u, u_history
    return u

if __name__ == "__main__":
    epsilon = 0.03
    t_end = 1
    Nt = 128
    Nx = 128
    a = 0
    b = 2 * np.pi
    current_path = os.getcwd()
    savepath = os.path.join(current_path, 'latex/figures')
    os.makedirs(savepath, exist_ok=True)

    # 求解并保存历史数据
    u_numeric, u_history = solve_Burger(epsilon, t_end, Nt, a=a, b=b, Nx=Nx, save_history=True)

    x = np.linspace(a, b, Nx, endpoint=False)-np.pi
    
    # 绘制最终时刻的解
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_numeric.real, 'o-')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.savefig(os.path.join(savepath, 'Q3_burgers_solution.png'))
    plt.show()

    # 绘制三维图 (x, t) -> u
    t = np.linspace(0, t_end, Nt+1)
    X, T = np.meshgrid(x, t)

    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面
    surf = ax.plot_surface(X, T, u_history, cmap='viridis', edgecolor='none', alpha=0.8)
    
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.savefig(os.path.join(savepath, 'Q3_burgers_3d.png'), dpi=150)
    plt.show()
    
    # 绘制等高线图
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    contour = ax2.contourf(X, T, u_history, levels=20, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    fig2.colorbar(contour, ax=ax2)
    
    plt.savefig(os.path.join(savepath, 'Q3_burgers_contour.png'), dpi=150)
    plt.show()