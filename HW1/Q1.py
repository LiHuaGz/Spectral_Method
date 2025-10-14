'''
solve ODE
'''

import numpy as np
from scipy.linalg import solve
from scipy.integrate import quad
from fast_methods import order2_sparse, order4_sparse, spectral_parallel
import os
import tqdm
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def use_academic_mpl_style():
    """
    Set a clean, publication-style matplotlib theme.
    """
    mpl.rcParams.update({
        # Figure & save
        'figure.figsize': (6.4, 4.0),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        # Fonts
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIX', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        # Text sizes
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        # Lines & markers
        'lines.linewidth': 1.8,
        'lines.markersize': 5.5,
        # Grid & spines
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _sanitize_filename(s: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in s)


def save_figure(fig: plt.Figure, base_name: str, folder: str = 'Q1_results/plots') -> str:
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, _sanitize_filename(base_name) + '.png')
    fig.savefig(filename)
    return filename


def compute_convergence_order(N_vals, errors, method='log-log'):
    """
    Compute numerical convergence order from N values and corresponding errors.
    
    Args:
        N_vals: array of grid sizes
        errors: array of corresponding errors
    
    Returns:
        convergence_order: estimated order of convergence (float)
    """
    # convert to double float
    N_vals = np.array(N_vals, dtype=np.float64)
    errors = np.array(errors, dtype=np.float64)

    # Filter out zero or very small errors to avoid log issues
    valid_idx = (errors > 1e-20) & np.isfinite(errors)
    if np.sum(valid_idx) < 3:
        return np.nan
    
    N_valid = np.array(N_vals)[valid_idx]
    err_valid = errors[valid_idx]

    if method == 'log-log':
        # Log-linear regression: log(error) = log(C) - order * log(N)
        log_N = np.log(N_valid)
        log_err = np.log(err_valid)

        # Linear regression
        A = np.vstack([log_N, np.ones(len(log_N))]).T
    else:
        # Log-id regression: error = C * exp(-order * N)
        A = np.vstack([N_valid, np.ones(len(N_valid))]).T
        log_err = np.log(err_valid)
    try:
        slope, _ = np.linalg.lstsq(A, log_err, rcond=None)[0]
        return -slope  # Negative because error ~ N^(-order)
    except:
        return np.nan

def order2(a, f, N=1000):
    '''
    use two-order finite difference to solve the ODE,
    -u'' + a*u = f, u(0) = u(pi) = 0
    '''
    x = np.linspace(0, np.pi, N)
    h = np.pi/(N-1)
    b = f(x)
    b[0] = b[-1] = 0
    # 向量化构造A矩阵
    A = np.zeros((N, N))
    A[0, 0] = 1
    A[-1, -1] = 1
    idx = np.arange(1, N-1)
    A[idx, idx-1] = -1/h**2
    A[idx, idx] = 2/h**2 + a
    A[idx, idx+1] = -1/h**2
    u = solve(A, b)
    return u

def order4(a, f, N=1000):
    '''
    use four-order compact finite difference to solve the ODE
    '''
    x = np.linspace(0, np.pi, N)
    h = np.pi/(N-1)
    # 向量化构造A矩阵
    A = np.zeros((N, N))
    A[0, 0] = 1
    A[-1, -1] = 1
    idx = np.arange(1, N-1)
    A[idx, idx-1] = (a - 12/(h**2))
    A[idx, idx] = (10*a + 24/(h**2))
    A[idx, idx+1] = (a - 12/(h**2))
    # 向量化构造b
    f_x = f(x)
    f_xm1 = np.roll(f_x, 1)
    f_xp1 = np.roll(f_x, -1)
    b = np.zeros_like(x)
    b[idx] = f_xm1[idx] + 10*f_x[idx] + f_xp1[idx]
    b[0] = b[-1] = 0
    u = solve(A, b)
    return u

def spectral(a, f, N=1000):
    '''
    use spectral method to solve the ODE
    '''
    x = np.linspace(0, np.pi, N)
    n_arr = np.arange(1, N)
    # 向量化计算u_qta
    def quad_vec(n):
        return quad(lambda xx: f(xx) * np.sin(n * xx), 0, np.pi, limit=N)[0]
    u_qta = np.array([quad_vec(n) for n in n_arr]) * 2 / ((n_arr**2 + a) * np.pi)
    # 向量化重构u(x)
    u = np.dot(u_qta, np.sin(np.outer(n_arr, x)))
    return u

def _sanitize_sheet_name(name: str) -> str:
    """
    Sanitize sheet name for Excel compatibility.
    Excel sheet names cannot contain: / \ ? * [ ] :
    and must be <= 31 characters
    """
    # Replace invalid characters
    invalid_chars = ['/', '\\', '?', '*', '[', ']', ':']
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Replace other problematic characters
    sanitized = sanitized.replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '_')
    
    # Ensure length <= 31 characters
    if len(sanitized) > 31:
        sanitized = sanitized[:28] + '...'
    
    return sanitized

if __name__ == "__main__":
    # Configure plotting style
    use_academic_mpl_style()

    # Control whether to show figures interactively; plots are always saved
    SHOW_FIGURES = False
    SAVE_DIR = 'Q1_results/plots'

    a = 1
    fs = [
        lambda x: np.sin(5*x),
        lambda x: np.exp(2*x),
        lambda x: np.where((x >= 0) & (x < np.pi/2), 1.0, 0.0)
    ]
    # Human-readable labels for the functions in fs (since lambda __name__ is '<lambda>')
    fn_labels = [
        'sin(5x)',
        'exp(2x)',
        '1 on [0, π/2), else 0'
    ]
    # Math-text for titles
    fn_labels_math = [
        r'\sin(5x)',
        r'e^{2x}',
        r'\mathbf{1}_{[0,\pi/2)}(x)'
    ]
    # Filenames friendly suffixes
    fn_file_labels = [
        'sin_5x',
        'exp_2x',
        'indicator_0_pi_over_2'
    ]
    u_reals = [
        lambda x: np.sin(5*x)*2/52,
        lambda x: (np.exp(2*np.pi)/3-np.exp(np.pi)/2+np.exp(-np.pi)/6)/(np.exp(np.pi)-np.exp(-np.pi))*(np.exp(x)-np.exp(-x)) - np.exp(2*x)/3 + np.exp(x)/2 - np.exp(-x)/6,
        lambda x: np.where((x >= 0) & (x < np.pi/2), (np.exp(np.pi)-np.exp(np.pi/2)+np.exp(-np.pi)-np.exp(-np.pi/2))/(2*(np.exp(np.pi)-np.exp(-np.pi)))*(np.exp(x)-np.exp(-x))-np.exp(x)/2-np.exp(-x)/2+1 , (np.exp(np.pi)-np.exp(np.pi/2)+np.exp(-np.pi)-np.exp(-np.pi/2))/(2*(np.exp(np.pi)-np.exp(-np.pi)))*(np.exp(x)-np.exp(-x))+0.5*(np.exp(x-np.pi/2)-np.exp(x)+np.exp(-x+np.pi/2)-np.exp(-x)))
    ]

    N_range = sorted(set(list(range(5, 501, 10))) | set(list(range(8, 504, 10))))
    # Storage for convergence results
    convergence_results = []
    # Storage for detailed calculation data for Excel export
    detailed_data = {}
    
    for i,f in enumerate(fs):
        error = []
        # Store detailed data for this function
        detailed_data[fn_labels[i]] = {
            'N': [],
            '2nd_Order_Error': [],
            '4th_Order_Error': [],
            'Spectral_Error': []
        }
        
        for N in tqdm.tqdm(N_range, desc=f'Processing f(x) = {fn_labels[i]}'):
            u_order2 = order2_sparse(a, f, N)
            u_order4 = order4_sparse(a, f, N)
            u_spectral = spectral_parallel(a, f, N)
            x = np.linspace(0, np.pi, N)
            u_real = u_reals[i](x)
            
            error_2nd = np.max(np.abs(u_real - u_order2))
            error_4th = np.max(np.abs(u_real - u_order4))
            error_spectral = np.max(np.abs(u_real - u_spectral))
            
            error.append((error_2nd, error_4th, error_spectral))
            
            # Store detailed data
            detailed_data[fn_labels[i]]['N'].append(N)
            detailed_data[fn_labels[i]]['2nd_Order_Error'].append(error_2nd)
            detailed_data[fn_labels[i]]['4th_Order_Error'].append(error_4th)
            detailed_data[fn_labels[i]]['Spectral_Error'].append(error_spectral)
            
        error = np.array(error)
        
        # Compute convergence orders for each method
        order_2nd = compute_convergence_order(N_range, error[:, 0], method='log-log')
        order_4th = compute_convergence_order(N_range, error[:, 1], method='log-log')
        if i != 0:
            order_spectral = compute_convergence_order(N_range, error[:, 2], method='log-log')
        else:
            order_spectral = compute_convergence_order(N_range, error[:, 2], method='log-id')
        
        # Store convergence results
        convergence_results.append({
            'Function': fn_labels[i],
            'Function_Math': fn_labels_math[i],
            '2nd_Order_FD_Convergence': order_2nd,
            '4th_Order_FD_Convergence': order_4th,
            'Spectral_Method_Convergence': order_spectral
        })
        
        print(f"\n=== Convergence orders for f(x) = {fn_labels[i]} ===")
        print(f"  2nd-order FD:    {order_2nd:.3f}" if not np.isnan(order_2nd) else "  2nd-order FD:    N/A")
        print(f"  4th-order FD:    {order_4th:.3f}" if not np.isnan(order_4th) else "  4th-order FD:    N/A")
        print(f"  Spectral method: {order_spectral:.3f}" if not np.isnan(order_spectral) else "  Spectral method: N/A")
        
        fig, ax = plt.subplots()

        # Choose marker frequency to avoid clutter
        #mark_step = max(1, len(list(N_range)) // 12)

        ax.plot(np.log(N_range), np.log(error[:, 0]), '-', marker='*', 
                    label='2nd-order FD')
        ax.plot(np.log(N_range), np.log(error[:, 1]), '-', marker='o', 
                    label='4th-order FD')
        ax.plot(np.log(N_range), np.log(error[:, 2]), '-', marker='s', 
                    label='Spectral method')

        ax.set_xlabel('log(N)')
        ax.set_ylabel('log(Error)')
        #ax.set_title(fr'$f(x) = {fn_labels_math[i]}$')

        # Legend with convergence orders
        legend_labels = [
            f'2nd-order FD',
            f'4th-order FD',
            f'Spectral method'
        ]
        
        # Update legend labels
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, 
                 frameon=True, framealpha=0.9, edgecolor='0.7', loc='upper right')

        fig.tight_layout()

        # Save high-DPI figure
        filebase = f'Q1_error_f_{fn_file_labels[i]}'
        out_path = save_figure(fig, filebase, folder=SAVE_DIR)
        print(f'Saved figure: {out_path}')

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)
    
    # Save convergence results to CSV
    results_dir = 'Q1_results'
    os.makedirs(results_dir, exist_ok=True)
    
    df_convergence = pd.DataFrame(convergence_results)
    csv_path = os.path.join(results_dir, 'Q1_convergence_orders.csv')
    df_convergence.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\nSaved convergence orders to: {csv_path}')
    
    # Save all data to Excel with multiple sheets
    excel_path = os.path.join(results_dir, 'Q1_complete_results.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Save convergence orders summary
        df_convergence.to_excel(writer, sheet_name='Convergence_Orders', index=False)
        
        # Save detailed data for each function
        for func_name, data in detailed_data.items():
            df_detailed = pd.DataFrame(data)
            # Add convergence order information to detailed sheet
            func_idx = fn_labels.index(func_name)
            conv_info = convergence_results[func_idx]
            
            # Create a summary at the top
            summary_data = {
                'Parameter': ['Function', 'Math Expression', '2nd Order FD Convergence', '4th Order FD Convergence', 'Spectral Method Convergence'],
                'Value': [conv_info['Function'], conv_info['Function_Math'], 
                         f"{conv_info['2nd_Order_FD_Convergence']:.6f}" if not np.isnan(conv_info['2nd_Order_FD_Convergence']) else 'N/A',
                         f"{conv_info['4th_Order_FD_Convergence']:.6f}" if not np.isnan(conv_info['4th_Order_FD_Convergence']) else 'N/A',
                         f"{conv_info['Spectral_Method_Convergence']:.6f}" if not np.isnan(conv_info['Spectral_Method_Convergence']) else 'N/A']
            }
            df_summary = pd.DataFrame(summary_data)
            
            # Create sanitized sheet name
            sheet_name = _sanitize_sheet_name(f'Details_{func_name}')
            
            # Write summary and data with some spacing
            df_summary.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
            df_detailed.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(df_summary)+2)
    
    print(f'\nSaved complete results to Excel: {excel_path}')
    
    print(f'\n=== Summary ===')
    print(f'All results saved to: {results_dir}/')
    print(f'CSV file: {csv_path}')
    print(f'Excel file: {excel_path}')
    print(f'Figures saved to: {SAVE_DIR}/')
