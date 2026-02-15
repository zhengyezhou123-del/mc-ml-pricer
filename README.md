# mc-ml-pricer
# Monte Carlo vs Machine Learning for European Call Pricing

## Overview

This project explores replacing Monte Carlo option pricing with fast
machine learning surrogates.

We compare three models:

1.  **Monte Carlo (baseline)**
2.  **Residual XGBoost model**
3.  **Residual Neural Network (PyTorch MLP)**

The goal is to approximate Monte Carlo prices with high accuracy while
achieving orders-of-magnitude faster inference.

------------------------------------------------------------------------

## Problem Setup

We price European call options with parameters:

-   ( S_0 ): spot price\
-   ( K ): strike\
-   ( T ): maturity\
-   ( r ): risk-free rate\
-   ( `\sigma `{=tex}): volatility

Monte Carlo simulation (with antithetic sampling and control variates)
is used to generate labels and standard errors.

------------------------------------------------------------------------

## Model 1 --- Monte Carlo (Baseline)

Monte Carlo estimator:

\[ P\_{MC} = e\^{-rT} `\frac{1}{N}`{=tex} `\sum`{=tex}\_{i=1}\^{N}
`\max`{=tex}(S_T\^{(i)} - K, 0) \]

**Properties** - Unbiased - Error scales as (O(1/`\sqrt{N}`{=tex})) -
Mean standard error ≈ **0.254** - Computationally expensive - Stochastic
output

------------------------------------------------------------------------

## Model 2 --- Improved Residual XGBoost

### Key Improvements

### 1️⃣ Residual Learning

Instead of predicting price directly:

\[ `\text{residual}`{=tex} = P\_{MC} - P\_{BS} \]

Final prediction:

\[ `\hat{P}`{=tex} = P\_{BS} + `\hat{\text{residual}}`{=tex} \]

Black--Scholes captures most structure, leaving a small smooth residual
to learn.

------------------------------------------------------------------------

### 2️⃣ Financial Feature Engineering

Added domain-aware features:

-   Moneyness ( S_0/K )
-   ( `\log`{=tex}(S_0/K) )
-   Intrinsic value
-   ( `\sqrt{T}`{=tex} ), ( `\log `{=tex}T )
-   Black--Scholes price
-   Black--Scholes Delta
-   Black--Scholes Vega

------------------------------------------------------------------------

### 3️⃣ Stabilized Noise-Aware Weighting

Instead of raw inverse variance weights:

\[ w = 1/`\text{stderr}`{=tex}\^2 \]

(which caused instability), we used bounded weights in a controlled
range.

This preserved noise-awareness while preventing domination by a few
low-noise samples.

------------------------------------------------------------------------

### 4️⃣ Regularization + Early Stopping

-   Shrinkage
-   Subsampling
-   Column subsampling
-   Early stopping

------------------------------------------------------------------------

### Results (XGBoost)

-   MAE: **0.212**
-   RMSE: **0.384**
-   RMSE / MC std: **1.51**

------------------------------------------------------------------------

## Model 3 --- Residual Neural Network (PyTorch)

### Architecture

-   3 hidden layers
-   GELU activations
-   BatchNorm
-   Dropout
-   AdamW optimizer

### Training Strategy

-   Same residual target
-   Same feature set
-   Same bounded weighting
-   Standardized inputs
-   Early stopping

------------------------------------------------------------------------

### Results (Neural Network)

-   MAE: **0.230**
-   RMSE: **0.390**
-   RMSE / MC std: **1.54**

------------------------------------------------------------------------

## Model Comparison

  Model                 MAE         RMSE          RMSE / MC Std
  --------------------- ----------- ------------- ---------------
  Monte Carlo           ---         0.254 (std)   1.00
  Residual XGBoost      **0.212**   **0.384**     1.51
  Residual Neural Net   0.230       0.390         1.54

------------------------------------------------------------------------

## Interpretation

-   ML error is \~1.5× Monte Carlo standard deviation.
-   ML inference is deterministic and near-instant.
-   Monte Carlo requires thousands of simulation paths per price.
-   Tree ensembles slightly outperform neural networks on tabular
    financial data.

------------------------------------------------------------------------

## Key Lessons

-   Residual learning is critical when analytic structure exists.
-   Financial feature engineering greatly improves stability.
-   Raw inverse-variance weighting can cause catastrophic overfitting.
-   Bounded noise-aware weighting stabilizes learning.
-   Tree ensembles remain highly competitive for structured tabular
    data.

------------------------------------------------------------------------
