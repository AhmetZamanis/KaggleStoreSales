# TIME SERIES REGRESSION PART 1 - KAGGLE STORE SALES COMPETITION


# TIME EFFECTS DECOMPOSITION MODEL SCRIPT

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set printing options
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format


# Set plotting options
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.autolayout"] = True
sns.set_theme(context="paper")


# Load data
df_train = pd.read_csv(
  "./ModifiedData/Final/train_modified_timefeats.csv", encoding="utf-8")


# Set datetime index
df_train = df_train.set_index(pd.to_datetime(df_train.date))
df_train = df_train.drop("date", axis=1)


# Aggregate total sales per day, across all stores and categories
total_sales = df_train.groupby("date").sales.sum()


# Create darts time series from total sales series
from darts import TimeSeries
ts = TimeSeries.from_series(total_sales, freq="D")


# Aggregate time features by mean
time_feats = df_train.drop(columns=['id', 'store_nbr', 'family', 'sales', 'onpromotion', 'transactions',
       'oil', 'city', 'state', 'store_type', 'store_cluster'], axis=1).groupby("date").mean(numeric_only=True)


# Add piecewise linear trend dummies
time_feats["trend"] = range(1, 1685)
time_feats.loc[time_feats.index=="2015-01-01"] # Knot at period 729
time_feats["trend_knot"] = 0
time_feats.iloc[728:,-1] = range(0, 956)
time_feats.loc[time_feats["trend"]>=729][["trend", "trend_knot"]] # Check start of knot


# Add Fourier features for monthly seasonality (weekly handled by day of week dummies)
from statsmodels.tsa.deterministic import DeterministicProcess
dp = DeterministicProcess(
  index = time_feats.index,
  constant = False,
  order = 0,
  seasonal = False,
  period = 28,
  fourier = 5 
)
time_feats = time_feats.merge(dp.in_sample(), how="left", on="date")


# Make Darts time series with time feats
ts_timefeats = TimeSeries.from_dataframe(
  time_feats, freq="D", fill_missing_dates=False)


# Define functions to perform log transformation and reverse it
def trafo_log(x):
  return x.map(lambda x: np.log(x+1))

def trafo_exp(x):
  return x.map(lambda x: np.exp(x)-1)


# Fill missing values (December 25)
from darts.dataprocessing.transformers import MissingValuesFiller
na_filler = MissingValuesFiller()
ts = na_filler.transform(ts)
ts_timefeats = na_filler.transform(ts_timefeats)


# Train-test split
y_train, y_val = trafo_log(ts[:-227]), trafo_log(ts[-227:])
x_train, x_val = ts_timefeats[:-227], ts_timefeats[-227:]


# Specify models: baseline, FFT and linear
from darts.models.forecasting.baselines import NaiveDrift, NaiveSeasonal
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.linear_regression_model import LinearRegressionModel

model_drift = NaiveDrift()
model_seasonal = NaiveSeasonal()

model_fft = FFT(
  nr_freqs_to_keep = 8,
  trend = "poly",
  trend_poly_degree = 1
)

model_linear = LinearRegressionModel(
  lags_future_covariates = [0],
  output_chunk_length = 15)


# Fit models on train data (pre-2017), predict validation data (2017)
model_drift.fit(y_train)
pred_drift = model_drift.predict(n = 227)

model_seasonal.fit(y_train)
pred_seasonal = model_seasonal.predict(n = 227)

model_fft.fit(y_train)
pred_fft = model_fft.predict(n = 227)

model_linear.fit(y_train, future_covariates = x_train)
pred_linear = model_linear.predict(n = 227, future_covariates = x_val)


# Define model scoring function
from darts.metrics import mape, rmse, rmsle
def perf_scores(val, pred, model="drift"):
  
  scores_dict = {
    "RMSE": rmse(trafo_exp(val), trafo_exp(pred)), 
    "RMSLE": rmse(val, pred), 
    "MAPE": mape(trafo_exp(val), trafo_exp(pred))
      }
      
  print("Model: " + model)
  
  for key in scores_dict:
    print(
      key + ": " + 
      str(round(scores_dict[key], 4))
       )
  print("--------")  


# Score models' performance
perf_scores(y_val, pred_drift, model="Naive drift")
perf_scores(y_val, pred_seasonal, model="Naive seasonal")
perf_scores(y_val, pred_fft, model="FFT")
perf_scores(y_val, pred_linear, model="Linear")


# FIG7: Plot models' predictions against actual values
fig7, axes7 = plt.subplots(2, sharex=True, sharey=True)
fig7.suptitle("Actual vs. predicted sales,\n time decomposition models")

# FFT
trafo_exp(y_val).plot(ax = axes7[0], label="Actual")
trafo_exp(pred_fft).plot(ax = axes7[0], label="Predicted")
axes7[0].set_title("FFT")

# Linear regression
trafo_exp(y_val).plot(ax = axes7[1], label="Actual")
trafo_exp(pred_linear).plot(ax = axes7[1], label="Predicted")
axes7[1].set_title("Linear regression")

# Show FIG7
plt.show()
fig7.savefig("./Plots/TimeModel/2017test.png", dpi=300)
plt.close("all")


# Backtest linear regression
# array([120080.4214,         nan,     31.611 ])
model_linear.backtest(
  ts, future_covariates = ts_timefeats, start = 365,
  forecast_horizon = 15, stride = 1, metric = [rmse, rmsle, mape],
  verbose = True)


# Retrieve historical forecasts for linear regression
pred_linear_hist = model_linear.historical_forecasts(
  ts, future_covariates = ts_timefeats, start = 365,
  forecast_horizon = 15, stride = 1,
  verbose = True)


# Plot historical forecasts for linear regression
ts.plot(label="Actual")
pred_linear_hist.plot(label="Predicted")
plt.title("Time decomposition linear model,\n historical forecasts")
plt.ylabel("sales")
plt.show()
plt.savefig("./Plots/TimeModel/LinearHistorical.png", dpi=300)
plt.close("all")


# Retrieve predictions and residuals for 2014-2017
res_linear = model_linear.residuals(
  trafo_log(ts), future_covariates = ts_timefeats, forecast_horizon = 1,
  verbose = True)
sales_decomped = res_linear[350:]
preds_time = trafo_log(ts[365:]) - res_linear[350:]


# Diagnose decomped sales innovation residuals
from darts.utils.statistics import plot_residuals_analysis, plot_pacf
plot_residuals_analysis(sales_decomped)
plt.show()
plt.savefig("./Plots/TimeModel/InnoResidsDiag.png", dpi=300)
plt.close("all")


# PACF plot of decomped sales residuals
plot_pacf(sales_decomped, max_lag=56)
plt.title("Partial autocorrelation plot,\n time decomposed sales")
plt.xlabel("Lags")
plt.ylabel("PACF")
plt.xticks(np.arange(0, 56, 10))
plt.xticks(np.arange(0, 56, 1), minor=True)
plt.grid(which='minor', alpha=0.5)
plt.grid(which='major', alpha=1)
plt.show()
plt.savefig("./Plots/TimeModel/PACFInnoResids.png", dpi=300)
plt.close("all")


# KPSS and ADF stationarity test on decomped sales residuals
from darts.utils.statistics import stationarity_test_kpss, stationarity_test_adf
stationarity_test_kpss(sales_decomped) # Null rejected, data is non-stationary
stationarity_test_adf(sales_decomped) # Null rejected, data is stationary around a constant


# Save decomposed sales for lags & covariates EDA
sales_decomped.pd_dataframe().to_csv(
  "./ModifiedData/Final/sales_decomped.csv", index=True, encoding="utf-8")


# Save predictions of model 1 to be added later to model 2 predictions
preds_time.pd_dataframe().to_csv(
  "./ModifiedData/Final/preds_model1.csv", index=True, encoding="utf-8")
