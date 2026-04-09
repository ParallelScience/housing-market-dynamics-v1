# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, zivot_andrews
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from arch.unitroot import PhillipsPerron

def run_stationarity_analysis():
    data_path = 'data/harmonized_housing_data.csv'
    df = pd.read_csv(data_path, index_col='observation_date', parse_dates=True)
    cols = ['CSUSHPINSA', 'HOUST', 'MORTGAGE30US']
    df = df[cols]
    results = []
    def format_row(items, widths):
        return " | ".join([str(item).ljust(width) for item, width in zip(items, widths)])
    print("--- Augmented Dickey-Fuller (ADF) Test ---")
    widths = [18, 10, 10, 8, 8, 8]
    print(format_row(["Series", "Test Stat", "p-value", "1%", "5%", "10%"], widths))
    print("-" * 75)
    for col in cols:
        adf_res = adfuller(df[col].dropna(), autolag='AIC')
        stat, pval, crit = adf_res[0], adf_res[1], adf_res[4]
        print(format_row([col, round(stat, 4), round(pval, 4), round(crit['1%'], 4), round(crit['5%'], 4), round(crit['10%'], 4)], widths))
        results.append("ADF " + col + " (Level): Stat=" + str(round(stat, 4)) + ", p-value=" + str(round(pval, 4)))
        adf_res_diff = adfuller(df[col].diff().dropna(), autolag='AIC')
        stat_d, pval_d, crit_d = adf_res_diff[0], adf_res_diff[1], adf_res_diff[4]
        diff_name = "Diff(" + col + ")"
        print(format_row([diff_name, round(stat_d, 4), round(pval_d, 4), round(crit_d['1%'], 4), round(crit_d['5%'], 4), round(crit_d['10%'], 4)], widths))
        results.append("ADF " + col + " (Diff): Stat=" + str(round(stat_d, 4)) + ", p-value=" + str(round(pval_d, 4)))
    print("\n--- Phillips-Perron (PP) Test ---")
    print(format_row(["Series", "Test Stat", "p-value", "1%", "5%", "10%"], widths))
    print("-" * 75)
    for col in cols:
        pp = PhillipsPerron(df[col].dropna())
        stat, pval, crit = pp.stat, pp.pvalue, pp.critical_values
        print(format_row([col, round(stat, 4), round(pval, 4), round(crit['1%'], 4), round(crit['5%'], 4), round(crit['10%'], 4)], widths))
        results.append("PP " + col + " (Level): Stat=" + str(round(stat, 4)) + ", p-value=" + str(round(pval, 4)))
        pp_diff = PhillipsPerron(df[col].diff().dropna())
        stat_d, pval_d, crit_d = pp_diff.stat, pp_diff.pvalue, pp_diff.critical_values
        diff_name = "Diff(" + col + ")"
        print(format_row([diff_name, round(stat_d, 4), round(pval_d, 4), round(crit_d['1%'], 4), round(crit_d['5%'], 4), round(crit_d['10%'], 4)], widths))
        results.append("PP " + col + " (Diff): Stat=" + str(round(stat_d, 4)) + ", p-value=" + str(round(pval_d, 4)))
    print("\n--- Johansen Cointegration Test ---")
    johansen_res = coint_johansen(df.dropna(), det_order=0, k_ar_diff=1)
    print("Trace Statistic:")
    j_widths = [5, 10, 10, 10, 10]
    print(format_row(["r", "Stat", "90%", "95%", "99%"], j_widths))
    print("-" * 55)
    for i in range(len(johansen_res.lr1)):
        r_name = "r<=" + str(i)
        print(format_row([r_name, round(johansen_res.lr1[i], 4), round(johansen_res.cvt[i, 0], 4), round(johansen_res.cvt[i, 1], 4), round(johansen_res.cvt[i, 2], 4)], j_widths))
        results.append("Johansen Trace r<=" + str(i) + ": Stat=" + str(round(johansen_res.lr1[i], 4)) + ", 95% CV=" + str(round(johansen_res.cvt[i, 1], 4)))
    print("\nMaximum Eigenvalue Statistic:")
    print(format_row(["r", "Stat", "90%", "95%", "99%"], j_widths))
    print("-" * 55)
    for i in range(len(johansen_res.lr2)):
        r_name = "r<=" + str(i)
        print(format_row([r_name, round(johansen_res.lr2[i], 4), round(johansen_res.cvm[i, 0], 4), round(johansen_res.cvm[i, 1], 4), round(johansen_res.cvm[i, 2], 4)], j_widths))
        results.append("Johansen MaxEig r<=" + str(i) + ": Stat=" + str(round(johansen_res.lr2[i], 4)) + ", 95% CV=" + str(round(johansen_res.cvm[i, 1], 4)))
    print("\n--- Zivot-Andrews Structural Break Test ---")
    za_widths = [15, 10, 10, 12, 8, 8, 8]
    print(format_row(["Series", "Test Stat", "p-value", "Break Date", "1%", "5%", "10%"], za_widths))
    print("-" * 85)
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    for idx, col in enumerate(cols):
        series = df[col].dropna()
        za_res = zivot_andrews(series, regression='c', autolag='AIC')
        stat, pval, crit, baselag, bpidx = za_res
        break_date = series.index[bpidx].strftime('%Y-%m-%d')
        print(format_row([col, round(stat, 4), round(pval, 4), break_date, round(crit['1%'], 4), round(crit['5%'], 4), round(crit['10%'], 4)], za_widths))
        results.append("ZA " + col + ": Stat=" + str(round(stat, 4)) + ", p-value=" + str(round(pval, 4)) + ", Break Date=" + break_date)
        y = series.values
        T = len(y)
        t_stats = np.full(T, np.nan)
        trim = 0.15
        t1 = int(T * trim)
        t2 = int(T * (1 - trim))
        dy = np.diff(y)
        y_lag = y[:-1]
        k = baselag
        if k > 0:
            X_lags = np.column_stack([dy[k-j : -j] for j in range(1, k+1)])
            dy_k = dy[k:]
            y_lag_k = y_lag[k:]
            time_trend = np.arange(k+2, T+1)
        else:
            dy_k = dy
            y_lag_k = y_lag
            time_trend = np.arange(2, T+1)
            X_lags = None
        for tb in range(t1, t2):
            DU = (time_trend > tb + 1).astype(float)
            X_cols = [np.ones(len(dy_k)), y_lag_k, time_trend, DU]
            X = np.column_stack(X_cols)
            if X_lags is not None:
                X = np.column_stack((X, X_lags))
            try:
                inv_XX = np.linalg.inv(X.T @ X)
                beta = inv_XX @ (X.T @ dy_k)
                resids = dy_k - X @ beta
                sigma2 = np.sum(resids**2) / (len(dy_k) - X.shape[1])
                var_beta = sigma2 * inv_XX
                se_alpha = np.sqrt(var_beta[1, 1])
                t_stats[tb] = beta[1] / se_alpha
            except Exception:
                pass
        ax = axes[idx]
        ax.plot(series.index, t_stats, label='Zivot-Andrews t-statistic', color='blue')
        ax.axhline(crit['5%'], color='red', linestyle='--', label='5% Critical Value')
        ax.axvline(series.index[bpidx], color='green', linestyle=':', label='Break Date: ' + break_date)
        ax.set_title('Zivot-Andrews Test Statistic over Time: ' + col)
        ax.set_ylabel('t-statistic')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'data/zivot_andrews_plot_2_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('\nZivot-Andrews plot saved to ' + plot_filename)
    summary_filename = 'data/stationarity_summary.txt'
    with open(summary_filename, 'w') as f:
        f.write('\n'.join(results))
    print('Stationarity summary saved to ' + summary_filename)

if __name__ == '__main__':
    run_stationarity_analysis()