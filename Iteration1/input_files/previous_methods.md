1. **Data Preprocessing and Seasonality Adjustment**
   - Standardize the temporal frequency to monthly. For the weekly `MORTGAGE30US`, calculate the monthly arithmetic mean.
   - Perform seasonal decomposition on all three series to check for residual seasonality. If detected, apply X-13ARIMA-SEATS or include seasonal dummy variables in the VAR specification to ensure shocks are not capturing seasonal lending patterns.
   - Trim all series to the common window (1987-01 to 2024-01).

2. **Stationarity and Structural Break Analysis**
   - Conduct ADF and PP tests to determine the order of integration. If series are $I(1)$, test for cointegration (Johansen test).
   - Test for structural breaks (e.g., 2008 crisis, COVID-19) using Zivot-Andrews or Chow tests. If significant breaks are identified, incorporate dummy variables into the VAR to account for regime shifts.
   - Decide between a VECM (if cointegrated) or a VAR in differences/levels based on unit root and cointegration results.

3. **VAR Model Specification and Exogeneity**
   - Define the vector $Y_t$ as [$\Delta(\text{MORTGAGE30US})$, $\Delta \log(\text{HOUST})$, $\Delta \log(\text{CSUSHPINSA})$].
   - Implement a Block-Exogenous structure where mortgage rates are treated as an exogenous block, ensuring they do not respond to contemporaneous shocks in housing starts or prices.
   - Determine the optimal lag length ($p$) using AIC, BIC, and HQIC, and perform diagnostic checks (Portmanteau for autocorrelation, Jarque-Bera for normality).

4. **Structural Identification via Sign Restrictions**
   - Apply sign restrictions on the impact matrix (1-month horizon) to avoid over-restricting the model.
   - Define 'Demand Shocks': positive impact on both $\Delta \log(\text{HOUST})$ and $\Delta \log(\text{CSUSHPINSA})$.
   - Define 'Supply Shocks': opposite signs for $\Delta \log(\text{HOUST})$ and $\Delta \log(\text{CSUSHPINSA})$.
   - Define 'Policy Shocks': an increase in mortgage rates leads to a decrease in both housing starts and prices.

5. **Estimation and Robustness**
   - Utilize the Rubio-Ramirez et al. (2010) algorithm to draw from the space of orthogonal matrices satisfying the sign restrictions.
   - Perform sensitivity analysis on the lag length $p$ (e.g., testing a range from 6 to 12 months) and the horizon of sign restrictions to ensure results are not artifacts of parameter selection.

6. **Impulse Response Function (IRF) Analysis**
   - Compute median IRFs and 68%/95% credible intervals for each structural shock.
   - Analyze the persistence and magnitude of the housing market's response to a one-standard-deviation shock in mortgage rates.

7. **Historical Decomposition**
   - Perform historical decomposition of the housing series to attribute fluctuations to Demand, Supply, or Mortgage Rate shocks.
   - Map these to key historical periods (2000s boom, 2008 bust, COVID-19) to validate the model against known market events.

8. **Forecast Error Variance Decomposition (FEVD)**
   - Calculate FEVD to quantify the proportion of forecast error variance attributable to each structural shock.
   - Assess the relative sensitivity of housing prices versus construction activity to financing costs to conclude the study.