1. **Data Preparation and Harmonization**
   - Aggregate `MORTGAGE30US` to monthly frequency using end-of-month values.
   - Construct the credit-market friction proxy: the spread between `MORTGAGE30US` and the 10-year Treasury constant maturity rate (DGS10).
   - Select `COMPUTSA` (New Privately-Owned Housing Units Completed) as the primary inventory/supply proxy to avoid the endogeneity issues inherent in the `MSACSR` ratio.
   - Apply log-transformations to `CSUSHPINSA`, `HOUST`, and `COMPUTSA` to stabilize variance; keep the mortgage-Treasury spread in levels (percentage points) to maintain interpretability.

2. **Model Specification and Estimation**
   - Estimate a Structural Vector Autoregression (SVAR) model. Given the increased variable set, employ a Bayesian VAR (BVAR) with Minnesota priors to mitigate the "curse of dimensionality" and ensure parameter stability.
   - Determine optimal lag length ($p$) using the Hannan-Quinn Information Criterion (HQIC).
   - Address heteroskedasticity by using robust standard errors (e.g., Newey-West). If residual autocorrelation persists, implement a VAR-GARCH-in-mean specification, with a fallback to standard robust estimation if convergence issues arise.

3. **Structural Identification via Sign Restrictions**
   - Implement the Rubio-Ramirez et al. (2010) algorithm to identify structural shocks.
   - Define 'Demand Shocks' as positive co-movement of `CSUSHPINSA` and `HOUST`.
   - Define 'Supply Shocks' as inverse co-movement of `CSUSHPINSA` and `HOUST`.
   - Define 'Financing Cost Shocks' as a positive shock to the mortgage-Treasury spread leading to a decline in both `CSUSHPINSA` and `HOUST`. Optionally apply zero-impact restrictions on the contemporaneous response of `HOUST` to the spread to isolate information effects.

4. **Regime-Based Comparative Analysis**
   - Split the sample into Pre-2008 (1987–2007) and Post-2012 (2012–2024) regimes.
   - Estimate the BVAR separately for these periods to compare the persistence of supply-side constraints.
   - Perform a Chow-type test on structural parameters to statistically validate shifts in market dynamics.

5. **Counterfactual Simulation (COVID-19 Era)**
   - Conduct a counterfactual simulation for 2020-2022 by treating the mortgage-Treasury spread as an exogenous input, fixed at 2019 average levels.
   - Acknowledge the "Lucas Critique" by noting the conditional nature of these results, assuming behavioral parameters remain constant.
   - Perform a sensitivity check by running the counterfactual using parameters from both the Pre-2008 and Post-2012 regimes to assess if the impact of financing costs on the COVID-era surge is regime-dependent.

6. **Robustness and Stability Testing**
   - Perform sensitivity analysis on sign restriction horizons (1-month vs. 3-month impact constraints).
   - Generate confidence bands for impulse response functions using bootstrapping (1,000 iterations).
   - Verify model stability through eigenvalue analysis of the companion matrix.

7. **Variance Decomposition and Interpretation**
   - Calculate the Forecast Error Variance Decomposition (FEVD) to quantify the relative importance of financing costs versus supply/demand shocks.
   - Explicitly document the "closed-system" limitations, acknowledging that identified "Supply Shocks" may partially capture unobserved demand-side factors like household formation or migration.

8. **Synthesis of Findings**
   - Aggregate results to construct a narrative on the evolution of housing market drivers.
   - Clearly distinguish between spread-driven shocks (credit risk/liquidity) and level-driven shocks (interest rate environment) in the post-2012 market equilibrium.