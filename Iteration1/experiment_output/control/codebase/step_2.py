# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

def main():
    data_dir = "data/"
    filepath = os.path.join(data_dir, "processed_housing_data.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    cols = ['d_MORTGAGE30US', 'd_HOUST_log', 'd_CSUSHPINSA_log']
    var_data = df[cols].dropna()
    print("--- VAR Lag Selection ---")
    model = VAR(var_data)
    best_aic = np.inf
    best_lag = 1
    for p in range(1, 13):
        res = model.fit(p)
        print("Lag " + str(p) + " - AIC: " + str(round(res.aic, 4)) + ", BIC: " + str(round(res.bic, 4)))
        if res.aic < best_aic:
            best_aic = res.aic
            best_lag = p
    print("\nSelected Optimal Lag based on AIC: " + str(best_lag))
    var_res = model.fit(best_lag)
    print("\n--- VAR Model Estimation ---")
    print("VAR Coefficients:")
    print(var_res.params)
    print("\n--- Portmanteau Test for Residual Autocorrelation ---")
    portmanteau = var_res.test_whiteness(nlags=best_lag + 12)
    print(portmanteau.summary())
    residuals = var_res.resid
    try:
        from arch import arch_model
    except ImportError:
        print("Module 'arch' not found. Please install it to run GARCH models.")
        raise
    garch_params = {}
    garch_cond_vol = pd.DataFrame(index=residuals.index, columns=residuals.columns)
    std_resid = pd.DataFrame(index=residuals.index, columns=residuals.columns)
    convergence_failed = False
    print("\n--- CCC-GARCH Estimation ---")
    for col in residuals.columns:
        scale = 100.0
        scaled_resid = residuals[col] * scale
        am = arch_model(scaled_resid, vol='Garch', p=1, q=1, mean='Zero')
        res = am.fit(disp='off')
        is_converged = (res.convergence_flag == 0)
        print("GARCH(1,1) for " + col + " convergence: " + str(is_converged))
        if not is_converged:
            convergence_failed = True
        garch_params[col] = res.params
        cond_vol = res.conditional_volatility / scale
        garch_cond_vol[col] = cond_vol
        std_resid[col] = residuals[col] / cond_vol
        print("Parameters for " + col + " (fitted on scaled residuals):")
        for param, val in res.params.items():
            print("  " + param + ": " + str(round(val, 6)))
    if convergence_failed:
        print("\nConvergence failed for at least one GARCH model. Falling back to Newey-West standard errors.")
    else:
        print("\nCCC-GARCH converged successfully.")
        R = std_resid.corr()
        print("\nConstant Conditional Correlation Matrix (R):")
        print(R)
        R.to_csv(os.path.join(data_dir, "ccc_garch_R.csv"))
        garch_cond_vol.to_csv(os.path.join(data_dir, "ccc_garch_vol.csv"))
    var_res.params.to_csv(os.path.join(data_dir, "var_params.csv"))
    residuals.to_csv(os.path.join(data_dir, "var_residuals.csv"))
    pd.DataFrame(var_res.sigma_u).to_csv(os.path.join(data_dir, "var_sigma_u.csv"))
    with open(os.path.join(data_dir, "var_lag.txt"), "w") as f:
        f.write(str(best_lag))
    print("\nModel parameters and residuals saved to data/ directory.")

if __name__ == '__main__':
    main()