# Data Description — U.S. Housing Market Dynamics (FRED)

## Source
Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis. All series are publicly available at https://fred.stlouisfed.org/.

## Files

### 1. `/home/node/work/projects/housing_v1/data/case_shiller.csv`
- **Series:** `CSUSHPINSA` — S&P/Case-Shiller U.S. National Home Price Index
- **Shape:** 470 rows × 2 columns (monthly, 1987-01 to ~2024)
- **Column `observation_date`:** end-of-month date (YYYY-MM-DD)
- **Column `CSUSHPINSA`:** composite price index, base = 2000-01-01 (index = 100). Seasonally adjusted. Unitless ratio scale — directly comparable over time; a value of 200 means prices are twice the 2000 baseline.
- **Note:** This is an index, not a dollar amount. It measures price changes, not absolute values.

### 2. `/home/node/work/projects/housing_v1/data/housing_starts.csv`
- **Series:** `HOUST` — Housing Starts: New Housing Units (Thousands, Monthly)
- **Shape:** 806 rows × 2 columns (monthly, 1959-01 to ~2024)
- **Column `observation_date`:** end-of-month date (YYYY-MM-DD)
- **Column `HOUST`:** thousands of new privately-owned housing units started in that month. Seasonally adjusted annual rate (SAAR). Units: thousands of housing units.
- **Note:** A housing "start" is defined as when excavation begins. Counts reflect the beginning of construction activity.

### 3. `/home/node/work/projects/housing_v1/data/mortgage_rates.csv`
- **Series:** `MORTGAGE30US` — 30-Year Fixed Rate Mortgage Average (% APY, Weekly)
- **Shape:** 2,872 rows × 2 columns (weekly, 1971-04-02 to ~2024)
- **Column `observation_date`:** date of the week ending (YYYY-MM-DD, typically a Friday)
- **Column `MORTGAGE30US`:** average offered rate on 30-year fixed-rate mortgages, committed by lenders, rounded to 2 decimal places. Units: percent per annum.

## Temporal Overlap
All three series overlap from 1987-01 to ~2024. The mortgage rate series is weekly ( Fridays); housing starts and Case-Shiller are monthly (end-of-month). For merged analysis, mortgage rates should be converted to monthly averages or end-of-month values.

## Suggested Research Directions
- Price-discovery dynamics: how do mortgage rate changes transmit to housing prices and construction activity?
- Lead-lag relationships between starts, prices, and financing costs
- Bubbles and busts: 2000s boom, 2008-2011 bust, COVID-era surge
- Volatility dynamics and asymmetric responses (rates up vs. rates down)

## Notes
- All series are published by FRED with no missing values in the downloaded period.
- The Case-Shiller index is revision-prone (late data points may shift slightly as more transactions close).
- Mortgage rate data is weekly; interpolation to monthly may introduce minor aggregation artifacts.
