### What is a time series?

In mathematics, a time series is a series of data points indexed in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data.Time series analysis is extensively used to forecast company sales, product demand, stock market trends, agricultural production etc.The fundamental idea for time series analysis is to decompose the original time series (sales, stock market trends, etc.) into several independent components.
 
The fundamental idea for time series analysis is to decompose the original time series (sales, stock market trends, etc.) into several independent components.

typically, business time series are divided into the following four components:

- Trend – overall direction of the series i.e. upwards, downwards etc.

- Seasonality – monthly or quarterly patterns.

- Cycle – long-term business cycles, they usually come after 5 or 7 years.

- Irregular remainder – random noise left after extraction of all the components

Interference of these components produces the final series.Why decomposing the original / actual time series into components?It is much easier to forecast the individual regular patterns produced through decomposition of time series than the actual series.

### Why decomposing the original / actual time series into components?

It is much easier to forecast the individual regular patterns produced through decomposition of time series than the actual series.

add images of Trend,Seasonality,Cyclical

### Stationary vs Non-Stationary Data

To effectively use ARIMA, we need to understand Stationarity in our data.

So what makes a data set Stationary?

- A Stationary series has constant mean and variance over time.

- A Stationary data set will allow our model to predict that the mean and variance will be the same in future periods.

Interference of these components produces the final series.

#### Stationary vs Non-Stationary Needs to be added
#### There are also mathematical tests you can use to test for stationarity in your data.

A common one is the Augmented Dickey–Fuller test

If we’ve determined your data is not stationary (either visually or mathematically), we will then need to transform it to be stationary in order to evaluate it and what type of ARIMA terms you will use.


One simple way to do this is through “differencing”.
Original Data
First Difference	Second Difference
#### differencing Images needs to be added
you can continue differencing until you reach stationarity (which you can check visually and mathematically).Each differencing step comes at the cost of losing a row of data.
 
For seasonal data, we can also difference by a season.

For example, if we had monthly data with yearly seasonality, we could difference by a time unit of 12, instead of just 1.
 
With our data now stationary it is time the p,d,q terms and how we choose them.

A big part of this are AutoCorrelation Plots and Partial AutoCorrelation Plots.
### Trends
From the plots it is obvious that there is some kind of increasing trend in the series along with seasonal variation.Stationarity is a vital assumption we need to verify if our time series follows a stationary process or not.

We can do by Plots: 
- review the time series plot of our data and visually check if there are any obvious trends or seasonality.
- Statistical tests: use statistical tests to check if the expectations of stationarity are met or have been violated.
## Trend Using Moving Average

### Moving Averages Over Time

- One way to identify a trend pattern is to use moving averages over a specific window of past observations.
- This smoothens the curve by averaging adjacent values over the specified time horizon (window).
### Seasonality

- People tend to go on vacation mainly during summer holidays.
- At some time periods during the year people tend to use aircrafts more frequently. We can check the hypothesis of a seasonal effect
### Noise
To understand the underlying pattern in the number of international airline passengers, we assume a multiplicative time series decomposition model Purpose is to understand underlying patterns in temporal data to use in more sophisticated analysis like Holt-Winters seasonal method or ARIMA.
Noise is the residual series left after removing the trend and seasonality components

### Stationarize a Time Series
Before models forecasting can be applied, the series must be transformed into a stationary time series.The Augmented-Dickey Fuller Test can be used to test whether or not a given time series is stationary.If the test statistic is smaller than the critical value, the hypothesis is rejected, the series would be stationary, and no further transformations of the data would be required.When the residuals (errors) in a time series are correlated with each other it is said to exhibit serial correlation.Autocorrelation is a better measurement for the dependency structure, because the autocovariacne will be affected by the underlying units of measurement for the observation.

### White Noise ACF & PACF
- Random process is white noise process
- Errors are serially uncorrelated if they are independent and identically distributed (iid).
- It is important because if a time series model is successful at capturing the underlying process, residuals of the model will be iid and resemble a white noise process.
 
### White Noise

Part of time series analysis is simply trying to fit a model to a time series such that the residual series is indistinguishable white noise.
### ACF & PACF

- The plots of the Autocorrelation function (ACF) and the Partial Autorrelation Function (PACF) are the two main tools to examine the time series dependency structure.
- The ACF is a function of the time displacement of the time series itself.
- It is the similarity between observations as a function of the time lag between them.
### PACF

The PACF is the conditional correlation between two variables under the assumptions that the effects of all previous lags on the time series are known.
### Random Walk

- What is special about the random walk is, that it is non-stationary, that is, if a given time series is governed by a random walk process it is unpredictable.
- It has high ACF for any lag length.
- The normal QQ plot and the histogram indicate that the series is not normally distributed.
- The random walk is a first order autoregressive process that is, this causes the process to be non-stationary.
- The process can be made stationary
### Auto Regressive Model – AR(p)

- The random walk process belongs to a more general group of processes, called autoregressive process.
- The current observation is a linear combination of past observations.
- An AR(1) time series is one period lagged weighted version of itself.
### The Moving Average Model - MA(q)

The moving average model MA(q) assumes that the observed time series can be represented by a linear combination of white noise error terms.The time series will always be stationary.
 
### ARIMA Forecasting

An autoregressive integrated moving average (ARIMA) model is an generalization of an autoregressive moving average (ARMA) model.Both of these models are fitted to time series data either to better understand the data or to predict future points in the series (forecasting).

ARIMA models are applied in some cases where data show evidence of non-stationarity, where an initial differencing step (corresponding to the "integrated" part of the model) can be applied one or more times to eliminate the non-stationarity.
There are three parameters $(p, d, q)$ that are used to parametrize ARIMA models. Hence, an ARIMA model is denoted as $ARIMA(p, d, q)$.Each of these three parts is an effort to make the time series stationary, i. e. make the final residual a white noise pattern.
 


### Box-Jenkins Approach to non-Seasonal ARIMA Modeling

In time series analysis, the Box–Jenkins method,named after the statisticians George Box and Gwilym Jenkins, applies autoregressive moving average (ARMA) or autoregressive integrated moving average (ARIMA) models to find the best fit of a time-series model to past values of a time series.The original model uses an iterative three-stage modeling approach:
Model identification and model selection:Making sure that the variables are stationary, identifying seasonality in the dependent series (seasonally differencing it if necessary), and using plots of the autocorrelation and partial autocorrelation functions of the dependent time series to decide which (if any) autoregressive or moving average component should be used in the model.Parameter estimation using computation algorithms to arrive at coefficients that best fit the selected ARIMA model.Model checking by testing whether the estimated model conforms to the specifications of a stationary univariate process.The residuals should be independent of each other and constant in mean and variance over time.
If the estimation is inadequate, we have to return to step one and attempt to build a better model.
To fit the time series data to a seasonal ARIMA model with parameters ARIMA(p, d, q)(P, D, Q)s$ the optimal parameters need to be found first.

This is done via grid search, the iterative exploration of all possible parameters constellations.Depending on the size of the model parameters $(p, d, q)(P, D, Q)s$ this can become an extremely costly task with regard to computation. We start of by generating all possible parameter constellation we'd like to evaluate.


### Akaike Information Criterion (AIC)

For all possible parameter constellations from both lists pdq and seasonal_pdq the alogrithm will create a model and eventually pick the best one to proceed.The best model is chosen based on the Akaike Information Criterion (AIC).The Akaike information criterion (AIC) is a measure of the relative quality of statistical models for a given set of data.Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models. Hence, AIC provides a means for model selection.It measures the trade-off between the goodness of fit of the model and the complexity of the model (number of included and estimated parameters).
### One step ahead prediction

The get_prediction and conf_int methods calculate predictions for future points in time for the previously fitted model and the confidence intervals associated with a prediction, respectively.The dynamic=False argument causes the method to produce a one-step ahead prediction of the time series.
### MSE

To quantify the accuracy between model fit and true observations we use the mean squared error (MSE).The MSE computes the squared difference between the true and predicted value.
### Out of sample Prediction

- To put the model to the real test with a 24-month-head prediction.

- This requires to pass the argument dynamic=False when using the get_prediction method.'''

### Long term forecasting

- Finally, a 10 year ahead forecast, leveraging a seasonal ARIMA model trained on the complete time series y.

- Grid search found the best model to be of form SARIMAX(2, 1, 3)(1, 2, 1)12 for the data vector y.

### Old techniques for time series.
Time series forecasting is a technique for the prediction of events through a sequence of time. It predicts future events by analyzing the trends of the past, on the assumption that future trends will hold similar to historical trends.
### Arima
An autoregressive integrated moving average, or ARIMA, is a statistical analysis model that uses time series data to either better understand the data set or to predict future trends. A statistical model is autoregressive if it predicts future values based on past values.
### Acf and Pcf
ACF is an (complete) auto-correlation function which gives us values of auto-correlation of any series with its lagged values. ACF considers all these components while finding correlations hence it's a 'complete auto-correlation plot'. PACF is a partial auto-correlation function.
### Time-Dependent Seasonal Components.
These components are defined as follows:

- Level: The average value in the series.
- Trend: The increasing or decreasing value in the series.
- Seasonality: The repeating short-term cycle in the series.
- Noise: The random variation in the series.
### Autoregressive(AR)
Autoregression is a time series model that uses observations from previous time steps as input to a regression equation to predict the value at the next time step. It is a very simple idea that can result in accurate forecasts on a range of time series problems.
### Moving average(MA) 
A moving average is defined as an average of fixed number of items in the time series which move through the series by dropping the top items of the previous averaged group and adding the next in each successive average.
### Mixed Arma- Modeler
I think the simplest way to look at it is to note that ARMA and similar models are designed to do different things than multi-level models, and use different data.

Time series analysis usually has long time series (possibly of hundreds or even thousands of time points) and the primary goal is to look at how a single variable changes over time. There are sophisticated methods to deal with many problems - not just autocorrelation, but seasonality and other periodic changes and so on.

Multilevel models are extensions from regression. They usually have relatively few time points (although they can have many) and the primary goal is to examine the relationship between a dependent variable and several independent variables. These models are not as good at dealing with complex relationships between a variable and time, partly because they usually have fewer time points (it's hard to look at seasonality if you don't have multiple data for each season).
### The random walk model
What Is the Random Walk Theory? Random walk theory suggests that changes in stock prices have the same distribution and are independent of each other. Therefore, it assumes the past movement or trend of a stock price or market cannot be used to predict its future movement.
### Box-jenkins methodology
The Box-Jenkins Model is a mathematical model designed to forecast data ranges based on inputs from a specified time series. The Box-Jenkins Model can analyze several different types of time series data for forecasting purposes. Its methodology uses differences between data points to determine outcomes.
### Forecasts with arima and var models.
- It can be said that ARIMA (0,1,2) model is the best model for forecasting Hessian sales price.

- ARIMA (3,2,2) model is the best model for forecasting Sacking sales price.

- VAR (4) model is the best model for forecasting the price of Carpet Backing Cloth (C.B.C).

- ARIMA (0,2,3) model is the best model for forecasting Others sales price.

Finally, it can be said that ARIMA model is the best model for forecasting the price of Jute Goods comparing with VAR model on the basis of Mean Absolute Percentage
Error (MAPE).The present study is an academic exercise for forecasting related issues. The findings of the present study can play a major role for further research as well as policy formulation to a great extent. Further research can be
conducted in future by using the deliberation of the present study.
Dynamic models with time-shifted explanatory variables.
### The koyck transformation
A device used to transform an infinite geometric lag model into a finite model with lagged dependent variable. While this makes estimation feasible, the transformed model is likely to have serial correlation in errors.
Partial adjustment and adaptive expectation models.
### Granger's causality tests?
The Granger causality test is a statistical hypothesis test for determining whether one time series is useful in forecasting another. If the probability value is less than any α level, then the hypothesis would be rejected at that level.
### Various approach to solve time series problem

- Autoregression(AR)

- Moving Average(MA)

- Autoregressive Moving Average(ARMA)

- Autoregressive Integrated Moving Average(ARIMA)

- Seasonal Autoregressive Integrated Moving-Average(SARIMA)

- Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors(SARIMAX)

- Vector Autoregression(VAR)

- Vector Autoregression Moving-Average(VARMA)

- Vector Autoregression Moving-Average with Exogenous Regressors(VARMAX)

- Simple Exponential Smoothing(SES)

- Holt Winter’s Exponential Smoothing(HWES)


### New Deep Learning based approach

- RNN
- LSTM
- GRU
### Time Series Model Performance Checking
- Mean Forcast Error
- Mean absolute Error
- Root Mean Squared Error
- Mean Squared error
### Complete end-to-end project with deployment(Prediction of nifty stock price and deployment)j
notebook given(convert into end to end)
•PH, a tractor and farm equipment manufacturing company, was established a few years after World War II.

•The company has shown a consistent growth in its revenue from tractor sales since its inception. However, over the years the company has struggled to keep it’s inventory and production cost down because of variability in sales and tractor demand.
 



•The management at PH is under enormous pressure from the shareholders and board to reduce the production cost.

•Additionally, they are also interested in understanding the impact of their marketing and farmer connect efforts towards overall sales.
 



•They have hired us as a data science and predictive analytics consultant.

•We will develop an ARIMA model to forecast sale / demand of tractor for next 3 years.

•Additionally, We will also investigate the impact of marketing program on sales by using an exogenous variable ARIMA model.
 



•An endogenous variable is one
that is influenced by other factors in the system. flower growth is affected by sunlight and is therefore endogenous.
Exogenous variables…
•are fixed when they enter the model.
•are taken as a “given” in the model.
•influence endogenous variables in the model.
•are not determined by the model.
•are not explained by the model.
 



•As a part of the project, one of the production units we are analyzing is based in South East Asia.

•This unit is completely independent and caters to neighboring geographies. This unit is just a decade and a half old. In 2014 , they captured 11% of the market share, a 14% increase from the previous year.
 



•However, being a new unit they have very little bargaining power with their suppliers to implement Just-in-Time (JiT) manufacturing principles that have worked really well in PH’s base location.

•Hence, they want to be on top of their production planning to maintain healthy business margins.
 



•Monthly sales forecast is the first step we have suggested to this unit towards effective inventory management.

•The MIS team shared the month on month (MoM) sales figures (number of tractors sold) for the last 12 years
