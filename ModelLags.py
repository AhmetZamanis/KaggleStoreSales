# TIME SERIES REGRESSION PART 1 - KAGGLE STORE SALES COMPETITION


# TIME EFFECTS LAGS & COVARIATES MODEL SCRIPT

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


# Load 13-17 original sales
from darts import TimeSeries
df_train = pd.read_csv(
  "./ModifiedData/Final/train_modified_timefeats.csv", encoding="utf-8")
df_train = df_train.set_index(pd.to_datetime(df_train.date))
df_train = df_train.drop("date", axis=1)
total_sales = df_train.groupby("date").sales.sum()
ts_orig = TimeSeries.from_series(total_sales, freq="D")
from darts.dataprocessing.transformers import MissingValuesFiller
na_filler = MissingValuesFiller()
ts_orig = na_filler.transform(ts_orig)


# Load 14-17 decomposed sales and covariates
sales_covars = pd.read_csv(
  "./ModifiedData/Final/sales_decomped_feats.csv", encoding="utf-8")


# Set datetime index
sales_covars = sales_covars.set_index(pd.to_datetime(sales_covars.date, dayfirst=True))
sales_covars = sales_covars.drop("date", axis=1)


# Create darts time series from decomposed sales sales series
ts = TimeSeries.from_series(sales_covars["sales"], freq="D")


# Make Darts time series with lags & covariates
ts_covars = TimeSeries.from_dataframe(
  sales_covars[["oil_ma336", "trns_ma7"]], freq="D", fill_missing_dates=False)


# Load 14-17 time preds, make darts series
preds_time = TimeSeries.from_csv(
  "./ModifiedData/Final/preds_model1.csv", time_col = "time", freq = "D")


# Define functions to perform log transformation and reverse it
def trafo_log(x):
  return x.map(lambda x: np.log(x+1))

def trafo_exp(x):
  return x.map(lambda x: np.exp(x)-1)


# Train-test split (pre-post 2017)
y_train, y_val = ts[:-227], trafo_log(ts_orig[-227:])
x_train, x_val = ts_covars[:-227], ts_covars[-227:]


# Scale covariates
from sklearn.preprocessing import StandardScaler
from darts.dataprocessing.transformers import Scaler
scaler = Scaler(StandardScaler())
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)


# Specify models: Baseline, ARIMA, linear, random forest
from darts.models.forecasting.baselines import NaiveDrift, NaiveSeasonal
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.models.forecasting.random_forest import RandomForest

model_drift = NaiveDrift()
model_seasonal = NaiveSeasonal()

model_arima = ARIMA(p = 1, d = 0, q = 0, trend = "n", random_state = 1923)

model_linear = LinearRegressionModel(
  lags = 1,
  lags_future_covariates = [0])
  
model_forest = RandomForest(
  lags = 1,
  lags_future_covariates = [0], 
  random_state = 1923,
  n_jobs = 3)


# Fit models on train data (pre-2017), predict validation data (2017)
model_drift.fit(y_train)
pred_drift = model_drift.predict(n = 227) + preds_time[-227:]

model_seasonal.fit(y_train)
pred_seasonal = model_seasonal.predict(n = 227) + preds_time[-227:]

model_arima.fit(y_train)
pred_arima = model_arima.predict(n = 227) + preds_time[-227:]

model_linear.fit(y_train, future_covariates = x_train)
pred_linear = model_linear.predict(n = 227, future_covariates = x_val) + preds_time[-227:]

model_forest.fit(y_train, future_covariates = x_train)
pred_forest = model_forest.predict(n = 227, future_covariates = x_val) + preds_time[-227:]


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
perf_scores(y_val, pred_arima, model="ARIMA")
perf_scores(y_val, pred_linear, model="Linear")
perf_scores(y_val, pred_forest, model="Random forest")


# FIG14: Plot models' predictions against actual values
fig14, axes14 = plt.subplots(3, sharex=True, sharey=True)
fig14.suptitle("Actual vs. predicted sales, hybrid models,\n black = Actual")

# ARIMA
trafo_exp(y_val).plot(ax = axes14[0], label="Actual")
trafo_exp(pred_arima).plot(ax = axes14[0], label="Predicted")
axes14[0].set_title("Linear regression + ARIMA")

# Linear regression
trafo_exp(y_val).plot(ax = axes14[1], label="Actual")
trafo_exp(pred_linear).plot(ax = axes14[1], label="Predicted")
axes14[1].set_title("Linear regression + Linear regression")

# Random forest
trafo_exp(y_val).plot(ax = axes14[2], label="Actual")
trafo_exp(pred_forest).plot(ax = axes14[2], label="Predicted")
axes14[2].legend("", frameon = False)
axes14[2].set_title("Linear regression + Random forest")

# Show fig14
plt.show()
fig14.savefig("./Plots/LagsModel/2017test.png", dpi=300)
plt.close("all")


# Go back to ModelTime's end

# Perform decomposition in Sklearn:
  # Fit, predict and retrieve residuals for 2013-2016 (train)
  # Predict and retrieve residuals for 2017 (valid)
  # Add them together and export them, use the preds as model 1 preds, and residuals as model 2 train

# Redo Lags EDA with Sklearn decomposed residuals

# Redo ModelLags with new features, Sklearn time preds and decomposed residuals





