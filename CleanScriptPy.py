#KAGGLE STORE SALES TIME SERIES FORECASTING - THIRD ATTEMPT


# PACKAGES and settings ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import darts
import sktime

np.set_printoptions(suppress=True, precision=8)
pd.options.display.float_format = '{:.8f}'.format




# LOAD MODIFIED DATA VERSION 2 ####
df_train = pd.read_csv("./ModifiedData/train_modified2.csv", encoding="utf-8")
df_test = pd.read_csv("./ModifiedData/test_modified2.csv", encoding="utf-8")


#set datetime index
df_train = df_train.set_index(pd.to_datetime(df_train.date))
df_train = df_train.drop("date", axis=1)
df_test = df_test.set_index(pd.to_datetime(df_test.date))
df_test = df_test.drop("date", axis=1)


#add category_store_no column for darts hierarchy
df_train["category_store_no"] = df_train["category"].astype(str) + "-" + df_train["store_no"].astype(str)
df_test["category_store_no"] = df_test["category"].astype(str) + "-" + df_test["store_no"].astype(str)


min(df_train["sales"]) 
#there is a negative sales number!

df_train.sales[df_train.sales<0]
#there are 165 negative sales numbers, all for 2013-12-25. interpolated values.
#replace them with zeroes.

df_train.sales[df_train.sales<0] = 0
#yes, that should work


#difference oil, onpromotion, transactions, with the train and test data combined
# df = pd.concat([df_train, df_test], axis=0)
# from sktime.transformations.series.difference import Differencer
# trafo_diff = Differencer(lags=1)
# df["oil"] = trafo_diff.fit_transform(df["oil"].values)
# df["onpromotion"] = trafo_diff.fit_transform(df["onpromotion"].values)
# df["transactions"] = trafo_diff.fit_transform(df["transactions"].values)
# df_train = df.iloc[0:3008016,]
# df_test = df.iloc[3008016:,]
# df_test = df_test.drop(columns=["sales", "transactions"], axis=1)
# del df



#check NAs
pd.isnull(df_train).sum()
pd.isnull(df_test).sum()
#no issue




#create wide dataframes with dates as rows, sales numbers for each hierarchy node as columns


#total
total = pd.DataFrame(
  data=df_train.groupby("date").sales.sum(),
  index=df_train.groupby("date").sales.sum().index)

#category
category = pd.DataFrame(
  data=df_train.groupby(["date", "category"]).sales.sum(),
  index=df_train.groupby(["date", "category"]).sales.sum().index)
category = category.reset_index(level=1)
category = category.pivot(columns="category", values="sales")

#store
store_no = pd.DataFrame(
  data=df_train.groupby(["date", "store_no"]).sales.sum(),
  index=df_train.groupby(["date", "store_no"]).sales.sum().index)
store_no = store_no.reset_index(level=1)
store_no = store_no.pivot(columns="store_no", values="sales")

#category_store_no
category_store_no = pd.DataFrame(
  data=df_train.groupby(["date", "category_store_no"]).sales.sum(),
  index=df_train.groupby(["date", "category_store_no"]).sales.sum().index)
category_store_no = category_store_no.reset_index(level=1)
category_store_no = category_store_no.pivot(columns="category_store_no", values="sales")


#merge all wide dataframes
from functools import reduce
wide_frames = [total, category, store_no, category_store_no]

ts_train = reduce(lambda left, right: pd.merge(
  left, right, how="left", on="date"), wide_frames)
  
del total, category, store_no, wide_frames, category_store_no




## CREATE TARGET TIME SERIES ####
ts_train = darts.TimeSeries.from_dataframe(
  ts_train, freq="D", fill_missing_dates=False)
  
  
#create and map hierarchy to target series

#lists of hierarchy nodes
categories = df_train.category.unique().tolist()
stores = df_train.store_no.unique().astype(str).tolist()

#empty dict
hierarchy_target = dict()

#categories to sales loop
for category in categories:
  hierarchy_target[category] = ["sales"]

#stores to sales loop
for store in stores:
  hierarchy_target[store] = ["sales"]

#category-store combos to category and stores
from itertools import product

for category, store in product(categories, stores):
  hierarchy_target["{}-{}".format(category, store)] = [category, store]
  
hierarchy_target
#yes_that should work

#map hierarchy to ts_train
ts_train = ts_train.with_hierarchy(hierarchy_target)
ts_train.hierarchy

del category, store






## CREATE TIME FEATURES SERIES ####

#list of 1870 dataframes:
    #piecewise trend dummy with knot at 2015-01-01
    #fourier features: period 28-order 14 and period 7-order 2
    #calendar features


from statsmodels.tsa.deterministic import DeterministicProcess


#total sales time features

#first aggregate all calendar cols by mean
time_feats = df_train.drop(columns=['category', 'store_no', 'store_cluster', 'store_type', 'city', 'state',
       'sales', 'transactions', "oil", "onpromotion"], axis=1).groupby("date").mean(numeric_only=True)

#add piecewise linear trend dummies, knot at 2015-01-01
time_feats["trend"] = range(1, 1689)
time_feats.loc[time_feats.index=="2015-01-01"]
#knot is at period 731
time_feats["trend_knot"] = 0
time_feats.iloc[730:,41] = range(0,958)
time_feats.loc[time_feats["trend"]>=731][["trend", "trend_knot"]]
#yes, that should work


#add fourier features (monthly and weekly)
dp = DeterministicProcess(
  index=time_feats.index,
  period=28,
  fourier=7
)
time_feats = time_feats.merge(dp.in_sample(), how="left", on="date")

dp = DeterministicProcess(
  index=time_feats.index,
  period=7,
  fourier=2
)
time_feats = time_feats.merge(dp.in_sample(), how="left", on="date")

time_feats.columns
#yes, that should work

#make it a darts time series, add to list of time feature sets
time_feats = darts.TimeSeries.from_dataframe(
  time_feats, freq="D", fill_missing_dates=False)
time_covariates = [time_feats]
del time_feats




#loop for category time features
for cat in df_train.category.unique():
  #first aggregate all calendar cols by mean
  time_feats = df_train.loc[df_train.category==cat].drop(columns=['store_no', 'store_cluster', 'store_type', 'city', 'state',
       'sales', 'transactions', "oil", "onpromotion"], axis=1).groupby("date").mean(numeric_only=True)

  #add piecewise linear trend dummies, knot at 2015-01-01
  time_feats["trend"] = range(1, 1689)
  time_feats["trend_knot"] = 0
  time_feats.iloc[730:,41] = range(0,958)
  #yes, that should work


  #add fourier features (monthly and weekly)
  dp = DeterministicProcess(
    index=time_feats.index,
    period=28,
    fourier=7
  )
  time_feats = time_feats.merge(dp.in_sample(), how="left", on="date")

  dp = DeterministicProcess(
  index=time_feats.index,
  period=7,
  fourier=2
  )
  time_feats = time_feats.merge(dp.in_sample(), how="left", on="date")

  #make it a darts time series, add to list of time feature sets
  time_feats = darts.TimeSeries.from_dataframe(
    time_feats, freq="D", fill_missing_dates=False)
  time_covariates.append(time_feats)
  del time_feats
del cat
len(time_covariates) 
#33 series, each with 60 components




#loop for store time features
for store in df_train.store_no.unique():
  #first aggregate all calendar cols by mean
  time_feats = df_train.loc[df_train.store_no==store].drop(columns=["store_no", 'category', 'store_cluster', 'store_type', 'city', 'state',
       'sales', 'transactions', "oil", "onpromotion"], axis=1).groupby("date").mean(numeric_only=True)

  #add piecewise linear trend dummies, knot at 2015-01-01
  time_feats["trend"] = range(1, 1689)
  time_feats["trend_knot"] = 0
  time_feats.iloc[730:,41] = range(0,958)
  #yes, that should work


  #add fourier features (monthly and weekly)
  dp = DeterministicProcess(
    index=time_feats.index,
    period=28,
    fourier=7
  )
  time_feats = time_feats.merge(dp.in_sample(), how="left", on="date")

  dp = DeterministicProcess(
  index=time_feats.index,
  period=7,
  fourier=2
  )
  time_feats = time_feats.merge(dp.in_sample(), how="left", on="date")

  #make it a darts time series, add to list of time feature sets
  time_feats = darts.TimeSeries.from_dataframe(
    time_feats, freq="D", fill_missing_dates=False)
  time_covariates.append(time_feats)
  del time_feats
del store
len(time_covariates) 
#88 series, 60 columns each




#loop for category:store time features
for cat_store in df_train.category_store_no.unique():
  #first aggregate all calendar cols by mean
  time_feats = df_train.loc[df_train.category_store_no==cat_store].drop(
    columns=["store_no", 'category', 'store_cluster', 'store_type', 'city', 'state',
       'sales', 'transactions', "oil", "onpromotion"], axis=1).groupby("date").mean(numeric_only=True)

  #add piecewise linear trend dummies, knot at 2015-01-01
  time_feats["trend"] = range(1, 1689)
  time_feats["trend_knot"] = 0
  time_feats.iloc[730:,41] = range(0,958)
  #yes, that should work


  #add fourier features (monthly and weekly)
  dp = DeterministicProcess(
    index=time_feats.index,
    period=28,
    fourier=7
  )
  time_feats = time_feats.merge(dp.in_sample(), how="left", on="date")

  dp = DeterministicProcess(
  index=time_feats.index,
  period=7,
  fourier=2
  )
  time_feats = time_feats.merge(dp.in_sample(), how="left", on="date")

  #make it a darts time series, add to list of time feature sets
  time_feats = darts.TimeSeries.from_dataframe(
    time_feats, freq="D", fill_missing_dates=False)
  time_covariates.append(time_feats)
  del time_feats
del cat_store
len(time_covariates) 
#1870 series, each with 60 components






## MODEL 1: TIME MODEL ####

#preprocessing steps:
  #log transform sales, at each node of the hierarchy separately
    #REMEMBER TO BACKTRANSFORM BEFORE SUMMING PREDICTIONS, PERFORMING RECONCILIATION ETC.
    

#log transform the target series
def trafo_log(x):
  return x.map(lambda x: np.log(x+0.0001))

def trafo_exp(x):
  return x.map(lambda x: np.exp(x)-0.0001)


y_train1 = trafo_log(ts_train)
#yes, that should work



#train-valid split pre-post 2017
y_train1, y_val1 = y_train1[:-227], y_train1[-227:]




#specify model 1: time features, horizon 15
from darts.models.forecasting.linear_regression_model import LinearRegressionModel

model_lm1 = LinearRegressionModel(
  lags=7,
  lags_future_covariates=[0],
  output_chunk_length=15
)
#had to include lag 7 of the target and future covariates because darts requires it




#fit model 1 on train data, predict on valid data, all components

#first fit it on the top node to initialize pred_lm1 series
model_lm1.fit(y_train1["sales"], future_covariates=time_covariates[0][:-227])
pred_lm1 = model_lm1.predict(future_covariates=time_covariates[0][-227:], n=227)


#then loop over all target components except first, to fit and predict them
for i in range(1, len(y_train1.components)):
  model_lm1.fit(y_train1[y_train1.components[i]], future_covariates=time_covariates[i][:-227])
  pred_comp = model_lm1.predict(future_covariates=time_covariates[i][-227:], n=227)
  pred_lm1 = pred_lm1.stack(pred_comp)

del pred_comp, i


#embed the target hierarchy to the predictions series
pred_lm1 = pred_lm1.with_hierarchy(hierarchy_target) 
pred_lm1
#yes, that should work




## MODEL 1 PERFORMANCE SCORES ####
from darts.metrics import mape, rmse, rmsle
from statistics import fmean, stdev


#function for total sales performance measures
def measures_summary_total(val, pred, subset):
  
  def measure_rmse_total(val, pred, subset):
    return rmse([trafo_exp(val[c]) for c in subset], [trafo_exp(pred[c]) for c in subset])

  def measure_rmsle_total(val, pred, subset):
    return rmsle([trafo_exp(val[c]) for c in subset], [trafo_exp(pred[c]) for c in subset])

  def measure_mape_total(val, pred, subset):
    return mape([trafo_exp(val[c]) for c in subset], [trafo_exp(pred[c]) for c in subset])

  scores_dict = {
    "RMSE": measure_rmse_total(val, pred, subset), 
    "RMSLE": measure_rmsle_total(val, pred, subset), 
    "MAPE": measure_mape_total(val, pred, subset)
      }
  
  for key in scores_dict:
    print(
      key + ": " + 
      str(round(scores_dict[key], 4))
       )
    print("--------")  
  



#function for bottom node sales performance measures
def measures_summary(val, pred, subset):
  
  def measure_rmse(val, pred, subset):
    return rmse([trafo_exp(val[c]) for c in subset], [trafo_exp(pred[c]) for c in subset])

  def measure_rmsle(val, pred, subset):
    return rmsle([trafo_exp(val[c]) for c in subset], [trafo_exp(pred[c]) for c in subset])

  def measure_mape(val, pred, subset):
    return mape([trafo_exp(val[c]) for c in subset], [trafo_exp(pred[c]) for c in subset])

  scores_dict = {
    "RMSE": measure_rmse(val, pred, subset), 
    "RMSLE": measure_rmsle(val, pred, subset), 
    "MAPE": measure_mape(val, pred, subset)
      }
  
  for key in scores_dict:
    print(
      key + ": mean=" + 
      str(round(fmean(scores_dict[key]), 4)) + 
      ", sd=" + 
      str(round(stdev(scores_dict[key]), 4)) + 
      ", min=" + str(round(min(scores_dict[key]), 4)) + 
      ", max=" + 
      str(round(max(scores_dict[key]), 4))
       )
    print("--------")


#total sales scores
measures_summary_total(y_val1, pred_lm1, ["sales"])
# RMSE: 97956.234
# --------
# RMSLE: 0.1355
# --------
# MAPE: 10.837
# --------


#category totals scores
measures_summary(y_val1, pred_lm1, categories)
# RMSE: mean=107315.7153, sd=434552.9557, min=13.5299, max=2496677.8247
# --------
# RMSLE: mean=1.4374, sd=1.5728, min=0.1151, max=4.287
# --------
# MAPE: mean=1.1580057936634151e+22, sd=6.452679987428214e+22, min=7.7777, max=3.708592795126658e+23
# --------
#RMSE is bit lower than total sales, but RMSLE is much higher. 
  #this means the category totals are severely underpredicted
#for some categories, RMSLE is very low, lower than total sales' RMSLE. for others, it's very high, up to 4+
  #this suggests the base model fits some categories' time features well, but not others
#MAPE is arbitrarily large for some categories, so it impacts the overall stats as well


#store totals scores
measures_summary(y_val1, pred_lm1, stores)
# RMSE: mean=5430764.0683, sd=23746682.2948, min=1131.7137, max=141569881.17
# --------
# RMSLE: mean=1.1077, sd=1.97, min=0.2524, max=7.0679
# --------
# MAPE: mean=2.992736706688129e+18, sd=9.698984534115981e+18, min=97.3142, max=4.127484277057098e+19
# --------
#RMSE is much larger than category and total sales, but RMSLE is lower than category sales
  #this means the store totals are less underpredicted compared to category totals
#minimum RMSLE is twice the RMSLE of total sales, at 0.25.
  #this suggests the base model doesn't fit the time features of any store well
#MAPE is still arbitrarily large, but less so than category totals


#category-store combo scores
measures_summary(y_val1, pred_lm1, categories_stores)
# RMSE: mean=58052.336, sd=789122.8055, min=0.0, max=17837894.0514
# --------
# RMSLE: mean=1.3191, sd=1.3321, min=0.0, max=6.6247
# --------
# MAPE: mean=5.948201339927377e+20, sd=3.3037459043881923e+21, min=50.9017, max=7.164406673067096e+22
# --------
#RMSLE is slightly lower than category totals, higher than store totals
  #this is expected as we'd expect the category dynamics to be similar in each store
#RMSE is 0 or very close to 0 for some category-store combos
  #this is misleading as the scale of sales may be very small for some category-store combos
#MAPE is still arbitrarily large, less than category totals, more than store totals




## MODEL 1 PLOTS ####


#plot 2017 actual vs. preds

#total sales
y_val1["sales"].plot(label="actual")
pred_lm1["sales"].plot(label="lm1 preds")
plt.title("total sales")
plt.show()
plt.close()


#select categories

#BREAD/BAKERY
y_val1["BREAD/BAKERY"].plot(label="actual")
pred_lm1["BREAD/BAKERY"].plot(label="lm1 preds")
plt.title("BREAD/BAKERY sales")
plt.show()
plt.close()

#CELEBRATION
y_val1["CELEBRATION"].plot(label="actual")
pred_lm1["CELEBRATION"].plot(label="lm1 preds")
plt.title("CELEBRATION sales")
plt.show()
plt.close()


#select stores

#54
y_val1["54"].plot(label="actual")
pred_lm1["54"].plot(label="lm1 preds")
plt.title("store 54 sales")
plt.show()
plt.close()

#14
y_val1["14"].plot(label="actual")
pred_lm1["14"].plot(label="lm1 preds")
plt.title("store 14 sales")
plt.show()
plt.close()


#same category in several stores

#BREAD/BAKERY-54, BREAD/BAKERY-14
y_val1["BREAD/BAKERY-54"].plot(label="actual")
pred_lm1["BREAD/BAKERY-54"].plot(label="lm1 preds")
plt.title("BREAD/BAKERY-54 sales")
plt.show()
plt.close()

y_val1["BREAD/BAKERY-14"].plot(label="actual")
pred_lm1["BREAD/BAKERY-14"].plot(label="lm1 preds")
plt.title("BREAD/BAKERY-14 sales")
plt.show()
plt.close()


#CELEBRATION-54, CELEBRATION-14
y_val1["CELEBRATION-54"].plot(label="actual")
pred_lm1["CELEBRATION-54"].plot(label="lm1 preds")
plt.title("CELEBRATION-54 sales")
plt.show()
plt.close()

y_val1["CELEBRATION-14"].plot(label="actual")
pred_lm1["CELEBRATION-14"].plot(label="lm1 preds")
plt.title("CELEBRATION-14 sales")
plt.show()
plt.close()




#see how nodes sum up before reconciliation


#create categories_stores object
categories_stores = []
for category, store in product(categories, stores):
  categories_stores.append("{}-{}".format(category, store))

#function to plot hierarchy sums
def plot_forecast_sums(pred_series, val_series=None):
    pred_series = trafo_exp(pred_series)
    val_series = trafo_exp(val_series)
    
    plt.figure(figsize=(10, 5))
    
    val_series["sales"].plot(label="actual total sales", alpha=0.3, color="red")
    
    pred_series["sales"].plot(label="predicted total sales", alpha=0.3, color="grey")
    sum([pred_series[r] for r in categories]).plot(label="sum of categories")
    sum([pred_series[r] for r in stores]).plot(label="sum of stores")
    sum([pred_series[t] for t in categories_stores]).plot(
        label="sum of categories_stores"
    )
    

    legend = plt.legend(loc="best", frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor("white")


#plot hierarchy sums for lm1
plot_forecast_sums(pred_lm1, y_val1)
plt.title("lm1 preds for each hierarchy node")
plt.show()
plt.close()




## LM1 DIAGNOSTICS ####


#get and inspect inno residuals of LM1 predictions
res_lm1 = y_val1 - pred_lm1


#time plot, distribution, acf
from darts.utils.statistics import plot_residuals_analysis
plot_residuals_analysis(res_lm1["sales"])
plt.show()
plt.close()
#seems fairly stationary, maybe slight downward trend in residuals
#ACF only fairly significant at lag 1, slightly at 3-7
  #declining sigmoidal pattern
#distribution of residuals normal except for few outliers


#pacf
from darts.utils.statistics import plot_pacf
plot_pacf(res_lm1["sales"], max_lag=48)
plt.show()
plt.close()
#PACF spike in lag 1, slightly significant 3, 5


#kpss test for stationarity
from darts.utils.statistics import stationarity_test_kpss as kpss
from darts.utils.statistics import stationarity_test_adf as adf
kpss_res = kpss(res_lm1["sales"])
kpss_res
#test stat 1.77, p val 0.01 or lower, data is non-stationary with a linear trend
adf_res = adf(res_lm1["sales"])
adf_res
#test stat -2.94, p value 0.04, data is differenced stationary




#LM1 DOES NOT MAKE THE 2017 PREDICTIONS STATIONARY
  #backtest and get residuals for 2014-2017, and try again?
  #or immediately go back to time covariates and change the trend?












## CREATE LAGS & COVARIATES SERIES ####


#list of 1870 covariates
    #sales ema7, 
    #oil lags 2, 3, 7, 8, oil ema54, 
    #onpromotion



#total sales covariates

#first aggregate onpromotion by sum
total_covar = df_train.groupby("date").aggregate(
  {"onpromotion":"sum"
  }
)


# #add sales ema7
# total_covar["sales_ema7"] = df_train.groupby("date").sales.sum().ewm(span=7).mean()
#   #can't do this at this point. this has to come after the decomposition.
#   #use lags 1-7 in model specification instead


#add oil features: lags 2, 3, 7, 8 and EMA54
oil_2 = df_train.groupby("date").oil.mean().shift(2).fillna(method="bfill", axis=0)
oil_3 = df_train.groupby("date").oil.mean().shift(3).fillna(method="bfill", axis=0)
oil_7 = df_train.groupby("date").oil.mean().shift(7).fillna(method="bfill", axis=0)
oil_8 = df_train.groupby("date").oil.mean().shift(8).fillna(method="bfill", axis=0)
oil_ema54 = df_train.groupby("date").oil.mean().ewm(span=54).mean()

total_covar["oil_2"] = oil_2
total_covar["oil_3"] = oil_3
total_covar["oil_7"] = oil_7
total_covar["oil_8"] = oil_8
total_covar["oil_ema54"] = oil_ema54


#difference the covariates
from sktime.transformations.series.difference import Differencer
trafo_diff = Differencer(lags=1)
total_covar = trafo_diff.fit_transform(total_covar)


#minmax scale the covariates? or leave this to after train-test split in darts?


#then make it a darts time series and add it to a list
total_covar = darts.TimeSeries.from_dataframe(
  total_covar, freq="D", fill_missing_dates=False)
all_covars = [total_covar]
del total_covar
#yes, that should work




## MODEL 2: COVARIATES MODEL ####


#preprocessing steps:
  #difference oil, onpromotion, if not done before
  #minmax scale oil, onpromotion







## RECONCILIATION ####


#remember to reverse log transformation of final predictions before scoring





## PREDICTION ON TEST DATA ####


#create time features and covariates series for test data


#preprocessing:
  #difference oil and onpromotion
  #minmax scale with scaler fitted on training data
  #apply the 2 models on the log scale, reverse transformations before reconciliation
  

#reconciliation
