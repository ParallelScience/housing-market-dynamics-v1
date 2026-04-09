# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy import stats

def portmanteau_test(resids, p, h=16):
    T, K = resids.shape
    C0 = np.dot(resids.T, resids) / T
    inv_C0 = np.linalg.inv(C0)
    Q = 0
    for j in range(1, h + 1):
        Cj = np.dot(resids[j:].T, resids[:-j]) / T
        term = np.trace(np.linalg.multi_dot([Cj.T, inv_C0, Cj, inv_C0]))
        Q += term / (T - j)
    Q *= (T ** 2)
    df = K**2 * h - 7 * p
    pval = stats.chi2.sf(Q, df)
    return Q, pval, df

def jarque_bera_multivariate(resids):
    T, K = resids.shape
    Sigma = np.dot(resids.T, resids) / T
    P = np.linalg.cholesky(Sigma)
    inv_P = np.linalg.inv(P)
    w = np.dot(resids, inv_P.T)
    b1 = np.mean(w**3, axis=0)
    b2 = np.mean(w**4, axis=0)
    s3 = (T / 6.0) * np.sum(b1**2)
    s4 = (T / 24.0) * np.sum((b2 - 3.0)**2)
    jb_stat = s3 + s4
    df = 2 * K
    pval = stats.chi2.sf(jb_stat, df)
    return jb_stat, pval, df

if __name__ == '__main__':
    data_path = 'data/harmonized_housing_data.csv'
    df = pd.read_csv(data_path, index_col='observation_date', parse_dates=True)
    df['d_MORTGAGE'] = df['MORTGAGE30US'].diff()
    df['d_log_HOUST'] = np.log(df['HOUST']).diff()
    df['d_log_CS'] = np.log(df['CSUSHPINSA']).diff()
    data = df[['d_MORTGAGE', 'd_log_HOUST', 'd_log_CS']].dropna()
    model = VAR(data)
    lag_res = model.select_order(maxlags=12)
    print('--- VAR Lag Order Selection ---')
    print(lag_res.summary())
    p = lag_res.aic
    print('\nSelected optimal lag length (AIC): ' + str(p))
    Y = data.values
    T_full, K = Y.shape
    X_list = []
    for i in range(1, p + 1):
        X_list.append(Y[p-i : T_full-i])
    X = np.column_stack(X_list)
    X = np.column_stack((np.ones(T_full - p), X))
    Y_trunc = Y[p:]
    T = Y_trunc.shape[0]
    idx_M = [0] + [1 + i*K for i in range(p)]
    X_M = X[:, idx_M]
    model_M = sm.OLS(Y_trunc[:, 0], X_M).fit()
    model_H = sm.OLS(Y_trunc[:, 1], X).fit()
    model_P = sm.OLS(Y_trunc[:, 2], X).fit()
    A = np.zeros((K, 1 + K*p))
    A[0, idx_M] = model_M.params
    A[1, :] = model_H.params
    A[2, :] = model_P.params
    resids = np.zeros((T, K))
    resids[:, 0] = model_M.resid
    resids[:, 1] = model_H.resid
    resids[:, 2] = model_P.resid
    Sigma = np.dot(resids.T, resids) / T
    h = max(16, p + 4)
    Q_stat, Q_pval, Q_df = portmanteau_test(resids, p, h=h)
    jb_stat, jb_pval, jb_df = jarque_bera_multivariate(resids)
    print('\n--- Diagnostic Checks ---')
    print('Portmanteau Test (h=' + str(h) + '):')
    print('  Test Statistic: ' + str(round(Q_stat, 4)))
    print('  Degrees of Freedom: ' + str(Q_df))
    print('  p-value: ' + str(round(Q_pval, 6)))
    print('\nMultivariate Jarque-Bera Test:')
    print('  Test Statistic: ' + str(round(jb_stat, 4)))
    print('  Degrees of Freedom: ' + str(jb_df))
    print('  p-value: ' + str(round(jb_pval, 6)))
    print('\n--- Estimated Block-Exogenous VAR Coefficients ---')
    print('Equation 1: d_MORTGAGE (Exogenous Block)')
    for i, coef in zip(idx_M, model_M.params):
        name = 'Intercept' if i == 0 else 'd_MORTGAGE(L' + str((i-1)//K + 1) + ')'
        print('  ' + name + ': ' + str(round(coef, 4)))
    out_path = 'data/var_model_parameters.npz'
    np.savez(out_path, A=A, Sigma=Sigma, resids=resids, p=p, Y_trunc=Y_trunc, data_cols=data.columns.values)
    print('\nModel parameters saved to ' + out_path)