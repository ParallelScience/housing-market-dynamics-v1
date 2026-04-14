# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
os.environ['OMP_NUM_THREADS'] = '2'
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def run_adf(series, name):
    result = adfuller(series.dropna())
    print('ADF Test for ' + name + ':')
    print('  Test Statistic: ' + str(round(result[0], 4)))
    print('  p-value: ' + str(round(result[1], 4)))
    print('  Critical Values: ' + ', '.join([str(k) + ': ' + str(round(v, 4)) for k, v in result[4].items()]))
    print('')

if __name__ == '__main__':
    cs_path = '/home/node/work/projects/housing_v1/data/case_shiller.csv'
    hs_path = '/home/node/work/projects/housing_v1/data/housing_starts.csv'
    mr_path = '/home/node/work/projects/housing_v1/data/mortgage_rates.csv'
    cs_df = pd.read_csv(cs_path)
    hs_df = pd.read_csv(hs_path)
    mr_df = pd.read_csv(mr_path)
    cs_df['observation_date'] = pd.to_datetime(cs_df['observation_date'])
    hs_df['observation_date'] = pd.to_datetime(hs_df['observation_date'])
    mr_df['observation_date'] = pd.to_datetime(mr_df['observation_date'])
    cs_df.set_index('observation_date', inplace=True)
    cs_df.index = cs_df.index.to_period('M')
    hs_df.set_index('observation_date', inplace=True)
    hs_df.index = hs_df.index.to_period('M')
    mr_df.set_index('observation_date', inplace=True)
    mr_monthly = mr_df.resample('ME').mean()
    mr_monthly.index = mr_monthly.index.to_period('M')
    merged_df = pd.concat([cs_df, hs_df, mr_monthly], axis=1, join='inner')
    merged_df = merged_df.loc['1987-01':]
    print('Merged DataFrame Shape: ' + str(merged_df.shape))
    print('Date Range: ' + str(merged_df.index.min()) + ' to ' + str(merged_df.index.max()))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print('\nDescriptive Statistics:\n' + str(merged_df.describe()))
    missing_vals = merged_df.isnull().sum()
    print('\nMissing Values:\n' + str(missing_vals))
    missing_count = missing_vals.sum()
    if missing_count == 0:
        print('\nVerification: No missing values found in the merged dataframe.')
    else:
        print('\nVerification: Found ' + str(missing_count) + ' missing values in the merged dataframe.')
    merged_df['CSUSHPINSA_log'] = np.log(merged_df['CSUSHPINSA'])
    merged_df['HOUST_log'] = np.log(merged_df['HOUST'])
    print('\n--- ADF Tests Before Differencing ---')
    run_adf(merged_df['CSUSHPINSA_log'], 'Log(CSUSHPINSA)')
    run_adf(merged_df['HOUST_log'], 'Log(HOUST)')
    run_adf(merged_df['MORTGAGE30US'], 'MORTGAGE30US')
    diff_df = merged_df[['CSUSHPINSA_log', 'HOUST_log', 'MORTGAGE30US']].diff()
    diff_df.columns = ['d_CSUSHPINSA_log', 'd_HOUST_log', 'd_MORTGAGE30US']
    print('--- ADF Tests After Differencing ---')
    run_adf(diff_df['d_CSUSHPINSA_log'], 'd_Log(CSUSHPINSA)')
    run_adf(diff_df['d_HOUST_log'], 'd_Log(HOUST)')
    run_adf(diff_df['d_MORTGAGE30US'], 'd_MORTGAGE30US')
    final_df = pd.concat([merged_df, diff_df], axis=1)
    output_path = 'data/processed_housing_data.csv'
    final_df.to_csv(output_path)
    print('Processed dataframe saved to ' + output_path)