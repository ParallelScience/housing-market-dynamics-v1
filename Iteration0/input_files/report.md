

Iteration 0:
### Summary: Structural Identification of Housing Market Shocks (1987–2024)

**Methodology & Data**
- **Data:** Monthly FRED series: `CSUSHPINSA` (Case-Shiller), `HOUST` (Starts), `MORTGAGE30US` (30-yr rate).
- **Preprocessing:** STL seasonal adjustment; all series $I(1)$. VAR specified in first differences ($\Delta \text{MORTGAGE30US}, \Delta \log \text{HOUST}, \Delta \log \text{CSUSHPINSA}$).
- **Model:** SVAR with block-exogeneity (Mortgage rates exogenous to housing). Lag order $p=4$ (AIC).
- **Identification:** Sign restrictions (Rubio-Ramirez 2010) on 1-month impact matrix. 57.5% acceptance rate.

**Key Findings**
- **Transmission Asymmetry:** Construction activity (Starts) reacts swiftly to shocks but is supply-constrained (50.7% of long-run variance). Home prices exhibit sluggish short-term response but are dominated by Policy Shocks in the long run (70.1% of variance at 36 months).
- **Historical Validation:** 
    - **2008 Bust:** Driven by negative Demand Shocks and supply overhang.
    - **COVID-19 Surge:** Dual-engine phenomenon; accommodative Policy/Demand shocks collided with severe supply-side constraints.
- **Structural Breaks:** Zivot-Andrews test identified a significant break in `HOUST` (Feb 2006).

**Limitations & Uncertainties**
- **Residuals:** Multivariate Portmanteau and Jarque-Bera tests reject Gaussian white-noise assumptions (fat tails/autocorrelation present).
- **Identification:** Results rely on the validity of sign restrictions; while robust to sampling, they do not fully capture non-linear regime shifts (e.g., ZLB periods).

**Future Directions**
- **Model Extension:** Incorporate non-linearities (e.g., Threshold VAR) to account for the ZLB period or regime-dependent responses to rate hikes vs. cuts.
- **Refinement:** Investigate the "Supply Shock" definition; current results suggest it may conflate construction activity with broader macroeconomic cooling.
- **Data:** Integrate inventory or permit data to better isolate "Supply" from "Demand" shocks.
        