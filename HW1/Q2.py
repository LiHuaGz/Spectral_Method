'''
compute the L^2 error and esssup error between f and I_Nf,
where I_Nf is the Fourier interpolant of f, and x in [0, 2pi].
'''

import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
from scipy.integrate import quad
import tqdm
import pandas as pd
import os
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter

# 全局绘图风格（更有“科研味”）：衬线字体、数学字体、较粗线条等
mpl.rcParams.update({
    'text.usetex': False,          # 关闭系统 LaTeX 依赖，使用内置 mathtext（更通用）
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'figure.titlesize': 16,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

def _pi_formatter():
    import numpy as _np
    pi = _np.pi
    def fmt(x, pos):
        # 容差判断
        if _np.isclose(x, 0):
            return r"$0$"
        if _np.isclose(x, 2*pi):
            return r"$2\pi$"
        # 写成 (k/2)*pi 形式 (这里主刻度为 pi/2)
        k = x / (pi/2)
        if _np.isclose(k, round(k)):
            k = int(round(k))
            if k == 2:
                return r"$\pi$"
            if k % 2 == 0:  # 偶数 k = 2m -> m*pi
                m = k//2
                if m == 1:
                    return r"$\pi$"
                return fr"${m}\pi$"
            else:  # 奇数 -> (2m+1)/2 * pi
                return fr"$\frac{{{k}\pi}}{{2}}$"
        # 其它情况不标注
        return ""
    return FuncFormatter(fmt)

def evaluate_scalar_function(func, points):
    """Evaluate a scalar-valued callable on an array of points."""
    pts = np.asarray(points)
    flat_pts = pts.reshape(-1)
    values = np.array([func(p) for p in flat_pts])
    return values.reshape(pts.shape)

def compute_L2_error(f, I_Nf, split_points=1000):
    def integrand(x):
        return np.abs(f(x) - I_Nf(x))**2
    error, _ = quad(integrand, 0, 2*np.pi, limit=split_points)
    return np.sqrt(error)

def compute_esssup_error(f, I_Nf, split_points=1000):
    x = np.linspace(0, 2*np.pi, split_points, endpoint=False)
    f_x = f(x)
    I_Nf_x = evaluate_scalar_function(I_Nf, x)
    error = np.max(np.abs(f_x - I_Nf_x))
    return error

def fourier_interpolant(f, N):
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    f_x = f(x)
    f_hat = fft(f_x) / N
    half_N = N // 2
    I_Nf = lambda y: (
        np.sum(f_hat[0:half_N] * np.exp(1j * np.arange(0, half_N) * y), axis=0)
        + np.sum(f_hat[half_N+1:N] * np.exp(1j * np.arange(-half_N+1, 0) * y), axis=0)
        + 0.5 * f_hat[half_N] * np.exp(1j * half_N * y)
        + 0.5 * f_hat[half_N] * np.exp(-1j * half_N * y)
    )
    return I_Nf

def plot_all_N_for_f(func, func_expr, N_values, errors):
    import os, math, re
    x = np.linspace(0, 2*np.pi, 1000)
    num = len(N_values)
    first_row_cols = math.ceil(num / 2)
    remaining = num - first_row_cols
    rows = 1 if remaining <= 0 else 2
    fig = plt.figure(figsize=(4.8*first_row_cols, 3.9*rows))
    grid_cols = first_row_cols * 2  # finer columns for centering
    # 加大 wspace/hspace 形成更宽的横向间距
    gs = fig.add_gridspec(rows, grid_cols, hspace=0.48, wspace=0.55)
    axes = []
    # First row (full or majority)
    for j in range(first_row_cols):
        axes.append(fig.add_subplot(gs[0, 2*j:2*j+2]))
    # Second row (center if partially filled)
    if remaining > 0:
        if remaining == first_row_cols:  # full second row
            for j in range(remaining):
                axes.append(fig.add_subplot(gs[1, 2*j:2*j+2]))
        else:
            total_needed = remaining * 2
            start = (grid_cols - total_needed) // 2
            for j in range(remaining):
                axes.append(fig.add_subplot(gs[1, start + 2*j : start + 2*j + 2]))

    # 配置刻度格式化器（自动 π 表达）
    pi_formatter = _pi_formatter()

    # Plot each N
    for idx, (ax, N) in enumerate(zip(axes, N_values)):
        I_Nf = fourier_interpolant(func, N)
        f_x = func(x)
        I_Nf_x = evaluate_scalar_function(I_Nf, x)
        ax.plot(x, f_x, label=r"$f(x)$", color='C0', linewidth=1.8)
        ax.plot(x, I_Nf_x.real, label=r"$I_N f(x)$", color='C3', linestyle='--', linewidth=1.6)
        ax.set_title(rf"$N={N}$")
        ax.set_xlabel(r"$x$")
        # Add y-label to first subplot of each row
        if idx == 0:
            ax.set_ylabel(r"$f(x)$")
        else:
            # compare y0 with previous; if different row and first in that row -> label
            if abs(ax.get_position().y0 - axes[idx-1].get_position().y0) > 1e-6:
                ax.set_ylabel(r"$f(x)$")
        ax.set_xlim(0, 2*np.pi)
        # 主刻度: 每 π/2, 自动格式化
        ax.xaxis.set_major_locator(MultipleLocator(np.pi/2))
        ax.xaxis.set_major_formatter(pi_formatter)
        # 次刻度: 每 π/4 (可选)
        ax.xaxis.set_minor_locator(MultipleLocator(np.pi/4))
        ax.tick_params(axis='x', which='minor', length=3, labelsize=0)
        ax.legend(loc='upper right', frameon=True, fancybox=True, edgecolor='gray')
        ax.grid(True, linestyle=':', alpha=0.5)
        L2_error, esssup_error = errors[idx]
        # 可选：误差文本（保持注释）
        # label = (rf"$\\|\\cdot\\|_{{L^2}}={L2_error:.2e}$" + "\n" + rf"$\\|\\cdot\\|_\\infty={esssup_error:.2e}$")
        # ax.text(0.02, 0.95, label, transform=ax.transAxes, fontsize=9, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # fig.suptitle(rf"$f(x) = {func_expr}$", fontsize=16)
    # 使用 subplots_adjust 而不是 tight_layout，避免警告并手动控制间距
    fig.subplots_adjust(top=0.82, left=0.07, right=0.98, bottom=0.10)
    plots_dir = './Q2_results/plots'
    os.makedirs(plots_dir, exist_ok=True)
    safe_expr = re.sub(r'[^a-zA-Z0-9_\-]', '_', func_expr)
    fig.savefig(f'{plots_dir}/{safe_expr}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    dir = './Q2_results'
    if os.path.exists(dir):
        import shutil
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)

    # 定义测试函数与用于标题的 LaTeX 表达式（mathtext 语法）
    # 第一个字符串用于打印/索引，第二个为标题显示（LaTeX 友好）
    f_list = [
    (lambda x: x/(6 + 4 * np.cos(x)), r"\frac{x}{6+4\cos x}"),
        (lambda x: np.sin(x), r"\sin x"),
        (lambda x: np.sin(4*x), r"\sin(4x)"),
        (lambda x: np.sin(16*x), r"\sin(16x)"),
        (lambda x: np.sin(64*x), r"\sin(64x)"),
        (lambda x: np.sin(256*x), r"\sin(256x)"),
    (lambda x: 1/(1+x**2), r"\frac{1}{1+x^2}"),
        (
            lambda x: np.piecewise(
                x,
                [ (0 <= x) & (x < np.pi/2),
                  (np.pi/2 <= x) & (x < 3*np.pi/2),
                  (3*np.pi/2 <= x) & (x <= 2*np.pi) ],
                [0, 1, 0]
            ),
            r"0, 0\leq x<\frac{\pi}{2}; 1, \frac{\pi}{2}\leq x<\frac{3\pi}{2}; 0, \frac{3\pi}{2}\leq x\leq 2\pi"
        )
    ]
    N_values = [16, 32, 64, 128, 256]
    split_points = 1000

    # 保存所有误差和所有图片
    all_errors = []
    func_names = []
    for func, func_expr in tqdm.tqdm(f_list, desc="Processing functions"):
        errors = []
        for N in N_values:
            I_Nf = fourier_interpolant(func, N)
            L2_error = compute_L2_error(func, I_Nf, split_points)
            esssup_error = compute_esssup_error(func, I_Nf, split_points)
            print(f'N={N}, Function {func_expr}: L2 Error = {L2_error:.6e}, Esssup Error = {esssup_error:.6e}')
            errors.append((L2_error, esssup_error))
        # 新增：保存所有图片
        plot_all_N_for_f(func, func_expr, N_values, errors)
        all_errors.append(errors)
        func_names.append(func_expr)

    # 构造DataFrame，列为多级：N_values下有L2_error和esssup_error
    columns = pd.MultiIndex.from_product([N_values, ['L2_error', 'esssup_error']], names=['N', 'ErrorType'])
    data = []
    for errors in all_errors:
        row = []
        for L2, esssup in errors:
            # 格式化为两位小数的科学计数法字符串
            row.extend([f"{L2:.2e}", f"{esssup:.2e}"])
        data.append(row)
    main_df = pd.DataFrame(data, index=func_names, columns=columns)

    # L2误差 DataFrame
    l2_data = []
    for errors in all_errors:
        l2_row = [f"{L2:.2e}" for L2, _ in errors]
        l2_data.append(l2_row)
    l2_df = pd.DataFrame(l2_data, index=func_names, columns=N_values)

    # esssup误差 DataFrame
    esssup_data = []
    for errors in all_errors:
        esssup_row = [f"{esssup:.2e}" for _, esssup in errors]
        esssup_data.append(esssup_row)
    esssup_df = pd.DataFrame(esssup_data, index=func_names, columns=N_values)

    # 保存到 CSV 和 Excel 文件
    main_df.to_csv(f'{dir}/Q2_results.csv')
    with pd.ExcelWriter(f'{dir}/Q2_results.xlsx') as writer:
        main_df.to_excel(writer, sheet_name='Main')
        l2_df.to_excel(writer, sheet_name='L2 Error')
        esssup_df.to_excel(writer, sheet_name='Essup Error')

    print('所有误差已保存到 Q2_results.csv 和 Q2_results.xlsx')