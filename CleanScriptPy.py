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






## PREPROCESSING: TIME MODEL ####


#preprocessing pipeline:
  #log transform sales
    #remember to backtransform before reconciliation

  
#create log transform and backtransform invertible mapper
from darts.dataprocessing.transformers.mappers import InvertibleMapper

def log_func(x):
  if x==0:
    return np.log(x+0.00000001)
  else:
    return np.log(x)


def exp_func(x):
  if np.exp(x)==0.00000001:
    return 0
  else:
    return np.exp(x)

trafo_log = InvertibleMapper(log_func, exp_func)

#log transform the target series
y_train1 = trafo_log.transform(ts_train)
#throws error: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()


# #log throws warnings. try box-cox instead
# from darts.dataprocessing.transformers import BoxCox
# trafo_boxcox = BoxCox()
# 
# #boxcox transform the target series
# y_train1 = trafo_boxcox.fit_transform(ts_train)
# #throwing errors.


#train-valid split pre-post 2017
y_train1, y_val1 = y_train1[:-227], y_train1[-227:]


#specify model 1: time features, horizon 15
from darts.models.forecasting.linear_regression_model import LinearRegressionModel

model_lm1 = LinearRegressionModel(
  lags=1,
  lags_future_covariates=[0],
  output_chunk_length=15
)
#had to include lag 1 of the target and future covariates because darts requires it


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

del test, trafo_boxcox, dp


#embed the target hierarchy to the predictions series
pred_lm1 = pred_lm1.with_hierarchy(hierarchy_target) 
pred_lm1
#yes, that should work


#see how nodes sum up before reconciliation

#create ca
categories_stores = []
for category, store in product(categories, stores):
  categories_stores.append("{}-{}".format(category, store))


def plot_forecast_sums(pred_series, val_series=None):
    pred_series = trafo_log.inverse_transform(pred_series)
    val_series = trafo_log.inverse_transform(val_series)
    
    plt.figure(figsize=(10, 5))
    
    sum([val_series[t] for t in categories_stores]).plot(
      label="actual categories_stores"
    )
    pred_series["sales"].plot(label="total", alpha=0.3, color="grey")
    sum([pred_series[r] for r in categories]).plot(label="sum of categories")
    sum([pred_series[r] for r in stores]).plot(label="sum of stores")
    sum([pred_series[t] for t in categories_stores]).plot(
        label="sum of categories_stores"
    )
    

    legend = plt.legend(loc="best", frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor("white")


plot_forecast_sums(pred_lm1, y_val1)
plt.title("unreconciled")
plt.show()
plt.close()
#it's messed up. why tho
  #because it's on the log scale. lol.
  #backtransforming the log trafo doesn't work, because:
  #ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()


#score predictions at each node of the hierarchy (after reversing log transformation)







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




## PREPROCESSING: COVARIATES MODEL ####


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
