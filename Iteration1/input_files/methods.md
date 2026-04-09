1. **Data Preparation and Alignment**
   - Aggregate weekly `MORTGAGE30US` and `DGS10` into monthly averages. Calculate the spread (MORTGAGE30US - DGS10).
   - Merge all series (CSUSHPINSA, HOUST, Inventory, Spread) into a single monthly dataframe.
   - Explicitly handle publication lags: align the Inventory series to ensure that the data used for a given month is only that which would have been available at the time, preventing look-ahead bias.
   - Apply log-transformations to non-stationary variables (Prices, Starts, Inventory) and compute first differences for all variables to ensure stationarity.

2. **Model Specification and Volatility Handling**
   - Specify a Structural Vector Autoregression (SVAR) model: [$\Delta \text{Spread}$, $\Delta \log(\text{HOUST})$, $\Delta \log(\text{CSUSHPINSA})$, $\Delta \log(\text{Inventory})$].
   - Select lag length ($p$) using AIC/BIC criteria, ensuring the Portmanteau test confirms no residual autocorrelation.
   - Implement a Constant Conditional Correlation (CCC)-GARCH model to account for heteroskedasticity. If convergence issues arise, fall back to a standard VAR with Newey-West robust standard errors.

3. **Structural Identification via Sign Restrictions**
   - Apply sign restrictions on the impact matrix (1-month horizon) using the Rubio-Ramirez et al. (2010) algorithm:
     - **Demand Shock:** $\Delta \log(\text{HOUST})$ and $\Delta \log(\text{CSUSHPINSA})$ move in the same direction.
     - **Supply Shock:** $\Delta \log(\text{HOUST})$ and $\Delta \log(\text{CSUSHPINSA})$ move in opposite directions.
     - **Financing Cost Shock:** $\Delta \text{Spread}$ increases, while $\Delta \log(\text{HOUST})$ and $\Delta \log(\text{CSUSHPINSA})$ decrease.
   - Enforce the Financing Cost shock as the only shock permitted to move the spread significantly on impact, potentially using a zero-restriction on the impact of the spread on starts/prices if necessary to prevent unrealistic immediate reactions.

4. **Sub-sample Comparative Analysis**
   - Partition the data into Pre-2008 (1987–2007) and Post-2012 (2012–2024) regimes.
   - Estimate the SVAR for each period to compare the persistence of supply shocks on price indices.
   - Conduct formal parameter stability tests (e.g., Chow test or structural break tests) to quantify shifts in supply constraints.

5. **Counterfactual Simulation (COVID-19 Era)**
   - Perform two counterfactual simulations for 2020-2022: one holding the *spread* constant at 2019 levels, and one holding the *nominal mortgage rate* constant at 2019 levels.
   - Set the innovation (shock) to the financing variable to the difference between the actual value and the 2019 baseline, allowing endogenous variables (Starts, Prices, Inventory) to evolve according to the estimated structural coefficients.

6. **Robustness and Sensitivity Testing**
   - Test sensitivity of the identification by varying the sign restriction horizon (1-month vs. 3-month).
   - Verify that the inclusion of the inventory variable and the volatility model successfully addresses residual autocorrelation and heteroskedasticity.
   - Assess if adding a "neutral" restriction on the Inventory variable for the Financing Cost shock improves model identification.

7. **Variance Decomposition and Interpretation**
   - Calculate the Forecast Error Variance Decomposition (FEVD) for the housing price index to quantify the relative contribution of financing costs versus demand/supply shocks.
   - Interpret results within the context of a closed-system model, acknowledging that findings represent internal market dynamics.

8. **Synthesis of Findings**
   - Aggregate impulse response functions (IRFs) and historical decompositions to narrate the evolution of housing market drivers.
   - Document the divergence between demand-led cycles and supply-constrained equilibria, highlighting shifts in supply shock persistence in the post-2012 era.