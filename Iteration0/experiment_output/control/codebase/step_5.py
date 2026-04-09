# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import time

def compute_fevd(all_Theta, horizons):
    N, max_horizon, K, _ = all_Theta.shape
    fevd_median = np.zeros((len(horizons), K, K))
    for idx, H in enumerate(horizons):
        Theta_H = all_Theta[:, :H, :, :]
        Theta_sq = Theta_H ** 2
        Theta_sq_sum = np.sum(Theta_sq, axis=1)
        MSE = np.sum(Theta_sq_sum, axis=2, keepdims=True)
        fevd = Theta_sq_sum / MSE
        fevd_med = np.percentile(fevd, 50, axis=0)
        fevd_med = fevd_med / np.sum(fevd_med, axis=1, keepdims=True)
        fevd_median[idx] = fevd_med
    return fevd_median

def compute_historical_decomposition(resids, B, accepted_A0):
    T, K = resids.shape
    p = B.shape[0]
    N = accepted_A0.shape[0]
    Phi = np.zeros((T, K, K))
    Phi[0] = np.eye(K)
    for h in range(1, T):
        for j in range(1, min(h, p) + 1):
            Phi[h] += B[j - 1] @ Phi[h - j]
    C_all = np.zeros((N, T, K, K))
    for i in range(N):
        A0 = accepted_A0[i]
        A0_inv = np.linalg.inv(A0)
        eps = resids @ A0_inv.T
        Theta = np.einsum('hkl,lj->hkj', Phi, A0)
        for k in range(K):
            for j in range(K):
                C_all[i, :, k, j] = np.convolve(eps[:, j], Theta[:, k, j])[:T]
    C_median = np.percentile(C_all, 50, axis=0)
    C_accum = np.cumsum(C_median, axis=0)
    return C_accum

if __name__ == '__main__':
    var_params = np.load('data/var_model_parameters.npz', allow_pickle=True)
    resids = var_params['resids']
    p = int(var_params['p'])
    struct_res = np.load('data/structural_identification_results.npz', allow_pickle=True)
    accepted_A0 = struct_res['accepted_A0']
    B = struct_res['B']
    all_Theta = struct_res['all_Theta']
    T, K = resids.shape
    horizons = [1, 6, 12, 24, 36]
    fevd_median = compute_fevd(all_Theta, horizons)
    shock_names = ['Policy Shock', 'Demand Shock', 'Supply Shock']
    var_names = ['Mortgage Rate', 'Housing Starts', 'Home Prices']
    print('--- Forecast Error Variance Decomposition (FEVD) ---')
    for k, var in enumerate(var_names):
        print('\nVariance Decomposition for ' + var + ':')
        header = 'Horizon | ' + ' | '.join([s.ljust(15) for s in shock_names])
        print(header)
        print('-' * len(header))
        for idx, H in enumerate(horizons):
            row = str(H).ljust(7) + ' | '
            row += ' | '.join([(str(round(fevd_median[idx, k, j]*100, 2)) + '%').rjust(15) for j in range(K)])
            print(row)
    C_accum = compute_historical_decomposition(resids, B, accepted_A0)
    df = pd.read_csv('data/harmonized_housing_data.csv', index_col='observation_date', parse_dates=True)
    df['d_MORTGAGE'] = df['MORTGAGE30US'].diff()
    df['d_log_HOUST'] = np.log(df['HOUST']).diff()
    df['d_log_CS'] = np.log(df['CSUSHPINSA']).diff()
    data = df[['d_MORTGAGE', 'd_log_HOUST', 'd_log_CS']].dropna()
    dates = data.index[p:]
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    colors = ['#d62728', '#1f77b4', '#2ca02c']
    y_labels = ['Cumulative Change (pp)', 'Cumulative Log Change', 'Cumulative Log Change']
    for k in range(K):
        ax = axes[k]
        pos_data = np.maximum(C_accum[:, k, :], 0)
        neg_data = np.minimum(C_accum[:, k, :], 0)
        bottom_pos = np.zeros(T)
        bottom_neg = np.zeros(T)
        for j in range(K):
            ax.bar(dates, pos_data[:, j], bottom=bottom_pos, width=25, color=colors[j])
            bottom_pos += pos_data[:, j]
            ax.bar(dates, neg_data[:, j], bottom=bottom_neg, width=25, color=colors[j])
            bottom_neg += neg_data[:, j]
        total_explained = np.sum(C_accum[:, k, :], axis=1)
        ax.plot(dates, total_explained, color='black', linewidth=1.5)
        ax.set_title('Historical Decomposition: ' + var_names[k])
        ax.set_ylabel(y_labels[k])
        ax.axhline(0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3)
        ax.axvspan(pd.to_datetime('2002-01-01'), pd.to_datetime('2006-12-01'), color='gray', alpha=0.2)
        ax.text(pd.to_datetime('2004-06-01'), 0.85, '2000s Boom', ha='center', transform=ax.get_xaxis_transform(), fontsize=10)
        ax.axvspan(pd.to_datetime('2007-12-01'), pd.to_datetime('2011-12-01'), color='red', alpha=0.1)
        ax.text(pd.to_datetime('2009-12-01'), 0.85, '2008 Bust', ha='center', transform=ax.get_xaxis_transform(), fontsize=10)
        ax.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2022-12-01'), color='orange', alpha=0.2)
        ax.text(pd.to_datetime('2021-07-01'), 0.85, 'COVID-19', ha='center', transform=ax.get_xaxis_transform(), fontsize=10)
    handles = [mpatches.Patch(color=colors[j], label=shock_names[j]) for j in range(K)]
    handles.append(mlines.Line2D([], [], color='black', linewidth=1.5, label='Total Explained'))
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    timestamp = str(int(time.time()))
    plot_filename = 'data/historical_decomposition_5_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('\nHistorical decomposition plot saved to ' + plot_filename)
    out_path = 'data/fevd_results.npz'
    np.savez(out_path, fevd_median=fevd_median, horizons=horizons, C_accum=C_accum, dates=dates)
    print('FEVD and Historical Decomposition results saved to ' + out_path)