### What is a Time Series?

In mathematics, a time series is a series of data points indexed in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus, it is a sequence of discrete-time data. Time series analysis is extensively used to forecast company sales, product demand, stock market trends, agricultural production, and more.

The fundamental idea for time series analysis is to decompose the original time series (sales, stock market trends, etc.) into several independent components.

Typically, business time series are divided into the following four components:

- **Trend:** The overall direction of the series, i.e., upwards, downwards, etc.

- **Seasonality:** Monthly or quarterly patterns.

- **Cycle:** Long-term business cycles, which usually occur after 5 or 7 years.

- **Irregular Remainder:** Random noise left after extraction of all the components.

The interference of these components produces the final series. But why decompose the original/actual time series into components?

**Formula for Decomposition:**

The decomposition of a time series can be expressed as:

$$
Original Time Series = Trend + Seasonality + Cycle + Irregular Remainder
$$

It is much easier to forecast the individual regular patterns produced through the decomposition of a time series than the actual series.

Time series analysis allows us to better understand and model complex data patterns over time, making it a valuable tool in various fields, including finance, economics, and agriculture.


### Why decomposing the original / actual time series into components?

It is much easier to forecast the individual regular patterns produced through decomposition of time series than the actual series.




-------
------

### Stationary vs Non-Stationary Data

To effectively use ARIMA, we need to understand Stationarity in our data.

So what makes a data set Stationary?

- A Stationary series has constant mean and variance over time.

- A Stationary data set will allow our model to predict that the mean and variance will be the same in future periods.

Interference of these components produces the final series.

#### There are also mathematical tests you can use to test for stationarity in your data.

A common one is the Augmented Dickey–Fuller test (ADF).

**Augmented Dickey–Fuller Test (ADF) Formula:**

The ADF test is used to determine whether a unit root is present in a time series dataset, which is an indicator of non-stationarity. The null hypothesis of the test is that the time series has a unit root (is non-stationary). The formula for the ADF test statistic is:

$$
ADF = \frac{\Delta y_t - \phi \cdot \Delta y_{t-1}}{s_{\Delta y_t}}
$$

Where:
- $\Delta y_t$ represents the differenced time series at time $t$.
- $\Delta y_{t-1}$ represents the differenced time series at time $t-1$.
- $\phi$ is a coefficient estimated by the test.
- $s_{\Delta y_t}$ is the standard error of the differenced time series.

If the ADF test statistic is less than the critical value from a table of critical values, the null hypothesis is rejected, indicating that the time series is stationary.

If we've determined your data is not stationary (either visually or mathematically), we will then need to transform it to be stationary in order to evaluate it and determine what type of ARIMA terms you will use.

One simple way to do this is through "differencing."

- **Original Data**
- **First Difference**
- **Second Difference**

#### Differencing

Differencing is a technique to transform a non-stationary time series into a stationary one. It involves subtracting the previous value from the current value to remove trends or seasonality.

**Differencing Formula:**

The differenced time series at time $t$ is obtained by subtracting the value at time $t-1$ from the value at time $t$:

$$
Difference_t = X_t - X_{t-1}
$$

You can continue differencing until you reach stationarity (which you can check visually and mathematically). Each differencing step comes at the cost of losing a row of data.

For seasonal data, we can also difference by a season.

For example, if we had monthly data with yearly seasonality, we could difference by a time unit of 12, instead of just 1.

With our data now stationary, it is time to consider the p, d, q terms and how we choose them.

A big part of this involves AutoCorrelation Plots and Partial AutoCorrelation Plots.

------------
-----------



# Moving Averages Over Time

Moving averages are a fundamental tool in time series analysis for identifying trends and patterns by smoothing out data over a specific time window. They provide a clearer view of underlying information by reducing noise.

**Formula for Simple Moving Average (SMA):**

The Simple Moving Average (SMA) at time $t$ is calculated as:

$$
SMA_t = \frac{X_{t-1} + X_t + X_{t+1} + \ldots + X_{t-n}}{n}
$$

Where:
- $SMA_t$ is the moving average at time $t$.
- $X_t$ represents the value at time $t$.
- $n$ is the number of data points in the moving window.

### Example:

Consider a small dataset of daily temperatures:

```
Day | Temperature (°C)
-----------------------
1   | 20
2   | 22
3   | 25
4   | 24
5   | 21
```

**Calculation of 3-Day Moving Average (SMA) for Day 3:**

To calculate the 3-day Simple Moving Average (SMA) for Day 3, we use the following formula:

$$
SMA_3 = \frac{22 + 25 + 24}{3} = \frac{71}{3} = 23.67
$$

Where:
- $SMA_3$ represents the moving average at Day 3.
- We sum the temperatures of the three previous days (Day 1, Day 2, and Day 3) and divide by 3, which is the size of our moving window.
- The result, 23.67 degrees Celsius, is the 3-day SMA for Day 3.

This moving average helps in smoothing the temperature data and provides a more stable representation of the temperature trend over the specified period.

---

# Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)

The Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) are crucial tools for examining the dependency structure in time series data.

**ACF (Autocorrelation Function):**

- ACF measures the similarity between observations as a function of the time lag between them.
- It quantifies the linear relationship between data points at different time lags.
- It helps identify the number of lags needed to capture the underlying autocorrelation structure.

**PACF (Partial Autocorrelation Function):**

- PACF is the conditional correlation between two variables under the assumption that the effects of all previous lags on the time series are known.
- It helps in determining the order of autoregressive (AR) terms in an ARIMA model.

Both ACF and PACF are essential in selecting appropriate model orders ($p$, $d$, $q$) for time series analysis.

**Formula for ACF:**

The ACF at lag $k$ is calculated as:

$$
ACF(k) = \frac{\text{Cov}(X_t, X_{t-k})}{\text{Var}(X_t)}
$$

Where:
- $ACF(k)$ is the autocorrelation at lag $k$.
- $\text{Cov}(X_t, X_{t-k})$ is the covariance between $X_t$ and $X_{t-k}$.
- $\text{Var}(X_t)$ is the variance of $X_t$.

**Formula for PACF:**

The PACF at lag $k$ is obtained from a regression of $X_t$ on $X_{t-1}$ to $X_{t-k}$:

$$
PACF(k) = \frac{\text{Cov}(X_t, X_{t-k} | X_{t-1}, X_{t-2}, \ldots, X_{t-(k-1)})}{\text{Var}(X_t | X_{t-1}, X_{t-2}, \ldots, X_{t-(k-1)})}
$$


### Example:

Consider a time series dataset representing daily stock prices. We want to analyze the ACF and PACF of this dataset to determine potential dependencies.

**ACF and PACF plots:**

In the ACF plot, we observe significant spikes at lags 1, 7, and 14, indicating correlations at these lags. In the PACF plot, there's a significant spike at lag 1, suggesting a potential AR(1) component in our ARIMA model. The lags 7 and 14 may imply seasonality in the data, leading to further analysis.

These plots guide the selection of appropriate terms for an ARIMA model:

- **Autoregressive (AR) Component ($p$):** We consider the lag at which the PACF plot cuts off or becomes insignificant. In this case, the PACF plot becomes insignificant after lag 1, suggesting an AR(1) component ($p=1$).

- **Differencing (Integration) Component ($d$):** We check if differencing is required to make the data stationary. If the original data is non-stationary, we observe the number of differences needed to achieve stationarity. For example, if differencing once makes the data stationary, then $d=1$.

- **Moving Average (MA) Component ($q$):** In the ACF plot, we look at the lags where there are significant spikes. In this case, lags 1, 7, and 14 have significant spikes. We consider the highest significant lag, which is lag 14, as a potential MA component. Therefore, $q=14$.

In summary, based on the ACF and PACF plots:

- We consider an ARIMA model with $p=1$ (AR(1)), $d=1$ (one difference for stationarity), and $q=14$ (MA(14)) as a starting point for our time series analysis. Further model refinement and diagnostics can be performed to ensure a good fit and predictive accuracy.

These plots play a critical role in determining the order of ARIMA terms and, ultimately, in building a suitable model for time series forecasting.


------------------
-----------------

### Auto Regressive Model – AR(p)

The Auto Regressive Model (AR) is a fundamental component of time series analysis. It is used to model and forecast time series data by capturing the dependency of an observation on its previous values.

#### Key Concepts:

- **AR Process:** The random walk process belongs to a more general group of processes called autoregressive processes. In an AR process, the current observation is a linear combination of past observations.

- **AR(p):** AR models are often denoted as AR(p), where 'p' represents the order of the autoregressive process. An AR(1) time series, for example, is a one-period-lagged weighted version of itself.

#### Formula for AR(p):

The AR(p) model can be represented as follows:

$$
X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \ldots + \phi_p X_{t-p} + \epsilon_t
$$

Where:
- $X_t$ is the value at time 't.'
- $c$ is a constant (intercept).
- $\phi_1, \phi_2, \ldots, \phi_p$ are the autoregressive coefficients.
- $X_{t-1}, X_{t-2}, \ldots, X_{t-p}$ are the lagged values of the time series.
- $\epsilon_t$ represents white noise or the error term.

In an AR model, the value at time 't' depends on its own past values up to 'p' periods back, weighted by the autoregressive coefficients.

---

### The Moving Average Model - MA(q)

The Moving Average Model (MA) is another crucial component of time series analysis. It models time series data by representing the observed values as a linear combination of past white noise error terms.

#### Key Concepts:

- **MA Process:** The moving average model assumes that the observed time series can be represented by a linear combination of white noise error terms. It focuses on the influence of past forecast errors on the current observation.

- **MA(q):** MA models are often denoted as MA(q), where 'q' represents the order of the moving average process. An MA(1) model, for example, considers one lag of the forecast error.

#### Formula for MA(q):

The MA(q) model can be represented as follows:

$$
X_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \ldots + \theta_q \epsilon_{t-q}
$$

Where:
- $X_t$ is the value at time 't.'
- $\mu$ is the mean or constant.
- $\epsilon_t, \epsilon_{t-1}, \epsilon_{t-2}, \ldots, \epsilon_{t-q}$ are white noise error terms at different lags.
- $\theta_1, \theta_2, \ldots, \theta_q$ are the moving average coefficients.

In an MA model, the value at time 't' depends on white noise error terms at various lags, weighted by the moving average coefficients.

---

### ARIMA Forecasting

The AutoRegressive Integrated Moving Average (ARIMA) model is a versatile time series forecasting tool that combines elements from both AR and MA models. ARIMA models are used to analyze and forecast time series data, particularly when the data exhibit non-stationarity.

#### Key Concepts:

- **ARIMA Model:** An ARIMA model is a generalization of an AutoRegressive Moving Average (ARMA) model. It can be applied to time series data to understand the data patterns or predict future points in the series (forecasting).

- **Non-Stationarity:** ARIMA models are suitable when data exhibit non-stationarity, which means the statistical properties of the data change over time.

- **Order $(p, d, q)$:** An ARIMA model is characterized by three parameters: $p$, $d$, and $q$. These parameters are used to make the time series stationary and are often written as ARIMA$(p, d, q)$.

#### Formula for ARIMA$(p, d, q)$:

The ARIMA model can be represented as follows:

$$
\text{ARIMA}(p, d, q):
X_t' = c + \phi_1 X_{t-1}' + \phi_2 X_{t-2}' + \ldots + \phi_p X_{t-p}' + \epsilon_t
$$

Where:
- $X_t'$ is the differenced time series (stationary series).
- $c$ is a constant (intercept).
- $\phi_1, \phi_2, \ldots, \phi_p$ are the autoregressive coefficients.
- $\epsilon_t$ represents white noise or the error term.

In ARIMA modeling, the original non-stationary time series $X_t$ is differenced $d$ times to create a stationary time series $X_t'$. Then, an ARMA model with order $(p, q)$ is applied to the differenced series to capture the autocorrelations and moving average effects.

ARIMA models are powerful tools for time series analysis and forecasting, especially when dealing with data that exhibits trends and seasonality.

---
--- 
### Approaches to Solve Time Series Problems:

1. **Autoregression (AR)**: A method that models future values based on past values.

2. **Moving Average (MA)**: A method that models future values as a weighted average of past white noise error terms.

3. **Autoregressive Moving Average (ARMA)**: A method that combines autoregressive and moving average components to model time series data.

4. **Autoregressive Integrated Moving Average (ARIMA)**: A method that integrates differencing into the ARMA model to handle non-stationary data.

5. **Seasonal Autoregressive Integrated Moving-Average (SARIMA)**: An extension of ARIMA that includes seasonal components to model seasonal time series data.

6. **Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)**: Incorporates additional exogenous variables along with seasonal ARIMA components for modeling.

7. **Vector Autoregression (VAR)**: Used when dealing with multiple time series that influence each other.

8. **Vector Autoregression Moving-Average (VARMA)**: Combines VAR and moving average components to model multivariate time series data.

9. **Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)**: Extends VARMA by including exogenous variables.

10. **Simple Exponential Smoothing (SES)**: A basic exponential smoothing method suitable for univariate time series forecasting.

11. **Holt Winter’s Exponential Smoothing (HWES)**: Adds seasonality and trend components to SES for more sophisticated forecasting.

###  Deep Learning-Based Approaches:

12. **Recurrent Neural Networks (RNN)**: Deep learning models designed to handle sequential data, including time series.

13. **Long Short-Term Memory (LSTM)**: A type of RNN with specialized memory cells for capturing long-term dependencies in time series data.

14. **Gated Recurrent Unit (GRU)**: Another type of RNN similar to LSTM but with a simplified architecture.

### Time Series Model Performance Checking:


- **Mean Absolute Error (MAE)**: Computes the average absolute difference between actual and predicted values, providing a measure of forecast accuracy.

- **Root Mean Squared Error (RMSE)**: Calculates the square root of the average squared differences between actual and predicted values. RMSE is commonly used to assess forecasting error.

- **Mean Squared Error (MSE)**: Measures the average of the squared differences between actual and predicted values, with a focus on penalizing large errors.

These performance metrics help evaluate the accuracy and reliability of time series forecasting models, ensuring their effectiveness in real-world applications.


### Variables

**Endogenous Variable:**
- An endogenous variable is a variable within a system or model that is influenced by other factors or variables within the same system.
- In other words, it is a variable whose behavior or changes are explained by the interactions and relationships with other variables in the system.
- Endogenous variables are the focus of analysis and are often the ones we aim to model and understand.

**Example of Endogenous Variable:**
- Consider the growth of a plant's height over time.
- In this case, the plant's height is the endogenous variable because it is influenced by various factors within the system, such as sunlight, water, soil nutrients, and temperature.
- Changes in sunlight, water availability, and other factors directly affect the plant's growth, making its height an endogenous variable.

**Exogenous Variables:**
- Exogenous variables, on the other hand, are external to the system or model being analyzed.
- These variables are considered to be "given" and are typically not explained by the model itself.
- Exogenous variables can influence endogenous variables within the system but are not determined by the relationships within the model.

**Characteristics of Exogenous Variables:**
- Exogenous variables are fixed when they enter the model, meaning their values are typically considered constant over the modeling period.
- They are external factors that may influence the behavior of endogenous variables.
- Exogenous variables are often used in time series modeling to account for external influences on the variable of interest.

**Example of Exogenous Variable:**
- Continuing with the plant growth example, consider the amount of sunlight the plant receives each day.
- Sunlight is an exogenous variable because it is external to the plant and its growth process.
- While sunlight influences the plant's height (an endogenous variable), it is not explained or determined by the plant's internal growth processes.

In summary, endogenous variables are those influenced by factors within the system or model, while exogenous variables are external factors that may influence endogenous variables but are not explained by the model itself. Both types of variables are important to consider in time series analysis and modeling to capture the full range of influences on the variable of interest.



