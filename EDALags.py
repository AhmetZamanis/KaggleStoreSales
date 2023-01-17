# TIME SERIES REGRESSION PART 1 - KAGGLE STORE SALES COMPETITION


# LAGS & COVARIATES EDA SCRIPT

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
  "./ModifiedData/Final/train_modified_timefeats.csv", encoding = "utf-8")


# Set datetime index
df_train = df_train.set_index(pd.to_datetime(df_train.date))
df_train = df_train.drop("date", axis=1)


# Load decomposed total sales
sales_decomposed = pd.read_csv(
  "./ModifiedData/Final/sales_decomped.csv", encoding = "utf-8")
sales_decomposed = sales_decomposed.set_index(pd.to_datetime(sales_decomposed.date))
sales_decomposed = sales_decomposed.drop("date", axis=1)
  
# Aggregate covariates from df_train: oil, onpromotion, transactions
covariates = df_train.groupby("date").agg(
 { "oil": "mean",
  "onpromotion": "sum",
  "transactions": "sum"})


# Left join decomposed sales and covariates
sales_covariates = sales_decomposed.merge(covariates, on = "date", how = "left")
pd.isnull(sales_covariates).sum()


# Difference the covariates
from sktime.transformations.series.difference import Differencer
diff = Differencer(lags = 1)
sales_covariates[
  ['oil', 'onpromotion', 'transactions']] = diff.fit_transform(
    sales_covariates[['oil', 'onpromotion', 'transactions']])


# Time interpolate missing values in covariates (christmas days)
sales_covariates = sales_covariates.interpolate(method = "time")


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sktime.utils.plotting import plot_lags


# FIG8: Sales ACF - PACF plot
fig8, axes8 = plt.subplots(2, sharex=True)
fig8.suptitle('Autocorrelation and partial autocorrelation,\n decomposed sales, up to 14 days')
plot_acf(sales_covariates["sales"], lags=np.arange(0, 15, 1, dtype=int), ax=axes8[0], marker=".")
plot_pacf(sales_covariates["sales"], lags=np.arange(0, 15, 1, dtype=int), ax=axes8[1], method="ywm", marker=".")
axes8[0].xaxis.set_ticks(np.arange(0, 15, 1, dtype=int), minor=True)
axes8[0].xaxis.set_ticks(np.arange(0, 15, 7, dtype=int))
axes8[0].grid(which='minor', alpha=0.5)
axes8[1].xaxis.set_ticks(np.arange(0, 15, 1, dtype=int), minor=True)
axes8[1].xaxis.set_ticks(np.arange(0, 15, 7, dtype=int))
axes8[1].grid(which='minor', alpha=0.5)

# Show fig8
plt.show()
fig8.savefig("./Plots/LagsEDA/SalesAcfPacf.png", dpi=300)
plt.close("all")


# FIG9: Sales lag scatterplots
fig9, axes9 = plot_lags(
  sales_covariates["sales"], lags = [1,2,3,4,5,6,7,8,9],
  suptitle = "Sales lags")
  
# Show fig9
plt.show()
fig9.savefig("./Plots/LagsEDA/SalesLags.png", dpi=300)
plt.close("all")


# Calculate 9-day exponential moving average of sales lags
sales_covariates["sales_ema9"] = sales_covariates["sales"].rolling(
  window = 9, min_periods = 1, center = False, win_type = "exponential").mean()


# Plot sales_ema9 vs sales  
sns.regplot(
  data = sales_covariates,
  x = "sales_ema9",
  y = "sales"
)
plt.title("Relationship of sales and 9-day\n exponential moving average of sales")
plt.show()
plt.savefig("./Plots/LagsEDA/SalesEma9.png", dpi=300)
plt.close("all")


# Compare correlations of sales with lag 1 and sales_ema9
from scipy.stats import pearsonr, spearmanr
pearsonr(sales_covariates["sales"], sales_covariates["sales"].shift(1).fillna(method="bfill")) # 0.85
spearmanr(sales_covariates["sales"], sales_covariates["sales"].shift(1).fillna(method="bfill")) # 0.86
pearsonr(sales_covariates["sales"], sales_covariates["sales_ema9"]) #0.7
spearmanr(sales_covariates["sales"], sales_covariates["sales_ema9"]) #0.72


# Drop sales_ema9
sales_covariates = sales_covariates.drop("sales_ema9", axis=1)


# Add oil moving averages
sales_covariates = sales_covariates.assign(
  oil_ma7 = lambda x: x["oil"].rolling(window = 7, min_periods = 1, center = False).mean(),
  oil_ma14 = lambda x: x["oil"].rolling(window = 14, min_periods = 1, center = False).mean(),
  oil_ma28 = lambda x: x["oil"].rolling(window = 28, min_periods = 1, center = False).mean(),
  oil_ma84 = lambda x: x["oil"].rolling(window = 84, min_periods = 1, center = False).mean(),
  oil_ma168 = lambda x: x["oil"].rolling(window = 168, min_periods = 1, center = False).mean(),
  oil_ma336 = lambda x: x["oil"].rolling(window = 336, min_periods = 1, center = False).mean(),
)


# FIG10: Regplots of oil moving averages & sales
fig10, axes10 = plt.subplots(3,2, sharey=True)
fig10.suptitle("Oil price change moving averages\n & decomposed sales")

# MA7
sns.regplot(
  ax = axes10[0,0],
  data = sales_covariates,
  x = "oil_ma7",
  y = "sales"
)
axes10[0,0].set_xlabel("weekly MA")
axes10[0,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(sales_covariates["oil_ma7"], sales_covariates["sales"])[0]
      ), xy=(.6, .9), xycoords="axes fraction",
    bbox=dict(alpha=0.5))

# MA14
sns.regplot(
  ax = axes10[0,1],
  data = sales_covariates,
  x = "oil_ma14",
  y = "sales"
)
axes10[0,1].set_xlabel("biweekly MA")
axes10[0,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(sales_covariates["oil_ma14"], sales_covariates["sales"])[0]
      ), xy=(.6, .9), xycoords="axes fraction",
    bbox=dict(alpha=0.5))

# MA28
sns.regplot(
  ax = axes10[1,0],
  data = sales_covariates,
  x = "oil_ma28",
  y = "sales"
)
axes10[1,0].set_xlabel("monthly MA")
axes10[1,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(sales_covariates["oil_ma28"], sales_covariates["sales"])[0]
      ), xy=(.6, .9), xycoords="axes fraction",
    bbox=dict(alpha=0.5))

# MA84
sns.regplot(
  ax = axes10[1,1],
  data = sales_covariates,
  x = "oil_ma84",
  y = "sales"
)
axes10[1,1].set_xlabel("quarterly MA")
axes10[1,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(sales_covariates["oil_ma84"], sales_covariates["sales"])[0]
      ), xy=(.6, .9), xycoords="axes fraction",
    bbox=dict(alpha=0.5))

# MA168
sns.regplot(
  ax = axes10[2,0],
  data = sales_covariates,
  x = "oil_ma168",
  y = "sales"
)
axes10[2,0].set_xlabel("semi-annual MA")
axes10[2,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(sales_covariates["oil_ma168"], sales_covariates["sales"])[0]
      ), xy=(.6, .9), xycoords="axes fraction",
    bbox=dict(alpha=0.5))

# MA336
sns.regplot(
  ax = axes10[2,1],
  data = sales_covariates,
  x = "oil_ma336",
  y = "sales"
)
axes10[2,1].set_xlabel("annual MA")
axes10[2,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(sales_covariates["oil_ma336"], sales_covariates["sales"])[0]
      ), xy=(.6, .9), xycoords="axes fraction",
    bbox=dict(alpha=0.5))

# Show FIG10
plt.show()
fig10.savefig("./Plots/LagsEDA/OilMAs.png", dpi=300)
plt.close("all")

# The extreme values are in the first rows, so they are only 1-day, 2-day, 3-day etc.
# averages, with very low values (-2.5, -1.5 etc.). Values of this magnitude are
# never repeated in MAs, so they misleadingly affect the correlation.


# FIG11: Regplots of oil moving averages & sales, without extreme MA values
fig11, axes11 = plt.subplots(3,2, sharey=True)
fig11.suptitle("Oil price change moving averages\n (without extreme values) & decomposed sales")

# Calculate MAs without min_periods = 1, replacing NAs with random dist
extreme_oil = sales_covariates.assign(
  oil_ma7 = lambda x: x["oil"].rolling(window = 7, center = False).mean(),
  oil_ma14 = lambda x: x["oil"].rolling(window = 14, center = False).mean(),
  oil_ma28 = lambda x: x["oil"].rolling(window = 28, center = False).mean(),
  oil_ma84 = lambda x: x["oil"].rolling(window = 84, center = False).mean(),
  oil_ma168 = lambda x: x["oil"].rolling(window = 168, center = False).mean(),
  oil_ma336 = lambda x: x["oil"].rolling(window = 336, center = False).mean(),
)

# MA7
sns.regplot(
  ax = axes11[0,0],
  data = extreme_oil,
  x = "oil_ma7",
  y = "sales"
)
axes11[0,0].set_xlabel("weekly MA")
axes11[0,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(extreme_oil["oil_ma7"], extreme_oil["sales"], nan_policy='omit')[0]), 
      xy=(.6, .9), xycoords="axes fraction",bbox=dict(alpha=0.5))

# MA14
sns.regplot(
  ax = axes11[0,1],
  data = extreme_oil,
  x = "oil_ma14",
  y = "sales"
)
axes11[0,1].set_xlabel("biweekly MA")
axes11[0,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(extreme_oil["oil_ma14"], extreme_oil["sales"], nan_policy='omit')[0]), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5))

# MA28
sns.regplot(
  ax = axes11[1,0],
  data = extreme_oil,
  x = "oil_ma28",
  y = "sales"
)
axes11[1,0].set_xlabel("monthly MA")
axes11[1,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(extreme_oil["oil_ma28"], extreme_oil["sales"], nan_policy='omit')[0]),
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5))

# MA84
sns.regplot(
  ax = axes11[1,1],
  data = extreme_oil,
  x = "oil_ma84",
  y = "sales"
)
axes11[1,1].set_xlabel("quarterly MA")
axes11[1,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(extreme_oil["oil_ma84"], extreme_oil["sales"], nan_policy='omit')[0]),
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5))

# MA168
sns.regplot(
  ax = axes11[2,0],
  data = extreme_oil,
  x = "oil_ma168",
  y = "sales"
)
axes11[2,0].set_xlabel("semi-annual MA")
axes11[2,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(extreme_oil["oil_ma168"], extreme_oil["sales"], nan_policy='omit')[0]),
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5))

# MA336
sns.regplot(
  ax = axes11[2,1],
  data = extreme_oil,
  x = "oil_ma336",
  y = "sales"
)
axes11[2,1].set_xlabel("annual MA")
axes11[2,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(extreme_oil["oil_ma336"], extreme_oil["sales"], nan_policy='omit')[0]),
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5))

# Show FIG11
plt.show()
fig11.savefig("./Plots/LagsEDA/OilMAsExtreme.png", dpi=300)
plt.close("all")






# Keep monthly oil MA, filling in NAs with random distribution
sales_covariates = sales_covariates.drop([
  "oil_ma7", "oil_ma14", "oil_ma84", "oil_ma168", "oil_ma336"], axis = 1)
sales_covariates["oil_ma28"] = sales_covariates["oil"].rolling(window = 28, center = False).mean()
rng = np.random.default_rng(1923)
mu = sales_covariates["oil_ma28"].mean()
sd = sales_covariates["oil_ma28"].std()
na_filler = pd.Series(rng.normal(loc=mu, scale=sd, size=len(sales_covariates["oil_ma28"])))
sales_covariates["oil_ma28"] = sales_covariates["oil_ma28"].fillna(na_filler)




# ONPROMOTION: A few lag plots, likely from 1-6 days before.
