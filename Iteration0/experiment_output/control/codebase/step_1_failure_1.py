# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import os
import time

if __name__ == '__main__':
    cs_path = '/home/node/work/projects/housing_v1/data/case_shiller.csv'
    hs_path = '/home/node/work/projects/housing_v1/data/housing_starts.csv'
    mr_path = '/home/node/work/projects/housing_v1/data/mortgage_rates.csv'
    df_cs = pd.read_csv(cs_path, parse_dates=['observation_date'], index_col='observation_date')
    df_hs = pd.read_csv(hs_path, parse_dates=['observation_date'], index_col='observation_date')
    df_mr = pd.read_csv(mr_path, parse_dates=['observation_date'], index_col='observation_date')
    df_cs_monthly = df_cs.resample('MS').first()
    df_hs_monthly = df_hs.resample('MS').first()
    df_mr_monthly = df_mr.resample('MS').mean()
    df = pd.concat([df_cs_monthly, df_hs_monthly, df_mr_monthly], axis=1)
    df = df.loc['1987-01-01':'2024-01-01']
    df = df.interpolate(method='linear')
    df_sa = pd.DataFrame(index=df.index)
    seasonality = pd.DataFrame(index=df.index)
    for col in df.columns:
        stl = STL(df[col], seasonal=13)
        res = stl.fit()
        seasonality[col] = res.seasonal
        df_sa[col] = df[col] - res.seasonal
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    axes[0].plot(df.index, df['CSUSHPINSA'], label='Raw', alpha=0.6)
    axes[0].plot(df_sa.index, df_sa['CSUSHPINSA'], label='Seasonally Adjusted', alpha=0.8, linestyle='--')
    axes[0].set_title('S&P/Case-Shiller U.S. National Home Price Index (CSUSHPINSA)')
    axes[0].set_ylabel('Index (Jan 2000 = 100)')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(df.index, df['HOUST'], label='Raw', alpha=0.6)
    axes[1].plot(df_sa.index, df_sa['HOUST'], label='Seasonally Adjusted', alpha=0.8, linestyle='--')
    axes[1].set_title('Housing Starts: New Housing Units (HOUST)')
    axes[1].set_ylabel('Thousands of Units')
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(df.index, df['MORTGAGE30US'], label='Raw', alpha=0.6)
    axes[2].plot(df_sa.index, df_sa['MORTGAGE30US'], label='Seasonally Adjusted', alpha=0.8, linestyle='--')
    axes[2].set_title('30-Year Fixed Rate Mortgage Average (MORTGAGE30US)')
    axes[2].set_ylabel('Percent (%)')
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'data/seasonal_adjustment_diagnostic_1_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('Diagnostic plot saved to ' + plot_filename)
    output_csv = 'data/harmonized_housing_data.csv'
    df_sa.to_csv(output_csv)
    print('Harmonized dataset saved to ' + output_csv)
    print('\n--- Harmonized Dataset Summary ---')
    print('Start Date: ' + str(df_sa.index.min().date()))
    print('End Date: ' + str(df_sa.index.max().date()))
    print('Number of observations: ' + str(len(df_sa)))
    print('\n--- Descriptive Statistics ---')
    print(df_sa.describe().to_string())
    print('\n--- First 5 rows ---')
    print(df_sa.head().to_string())
    print('\n--- Last 5 rows ---')
    print(df_sa.tail().to_string())