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
sales_decomposed = sales_decomposed.set_index(pd.to_datetime(sales_decomposed.date, dayfirst=True))
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
from scipy.stats import pearsonr, spearmanr


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


# Calculate 7-day exponential moving average of sales lags
sales_covariates["sales_ema7"] = sales_covariates["sales"].rolling(
  window = 7, min_periods = 1, center = False, win_type = "exponential").mean()


# Plot sales_ema7 vs sales  
sns.regplot(
  data = sales_covariates,
  x = "sales_ema7",
  y = "sales"
)
plt.title("Relationship of sales and 7-day\n exponential moving average of sales")
plt.show()
plt.savefig("./Plots/LagsEDA/SalesEma7.png", dpi=300)
plt.close("all")


# Compare correlations of sales with lag 1 and sales_ema8
pearsonr(sales_covariates["sales"], sales_covariates["sales"].shift(1).fillna(method="bfill")) # 0.79
spearmanr(sales_covariates["sales"], sales_covariates["sales"].shift(1).fillna(method="bfill")) # 0.80
pearsonr(sales_covariates["sales"], sales_covariates["sales_ema7"]) #0.68
spearmanr(sales_covariates["sales"], sales_covariates["sales_ema7"]) #0.68


# FIG10: Regplots of oil moving averages & sales
fig10, axes10 = plt.subplots(3,2, sharey=True)
fig10.suptitle("Oil price change moving averages\n & decomposed sales")

# Calculate MAs without min_periods = 1
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
  ax = axes10[0,0],
  data = extreme_oil,
  x = "oil_ma7",
  y = "sales"
)
axes10[0,0].set_xlabel("weekly MA")
axes10[0,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(extreme_oil["oil_ma7"], extreme_oil["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# MA14
sns.regplot(
  ax = axes10[0,1],
  data = extreme_oil,
  x = "oil_ma14",
  y = "sales"
)
axes10[0,1].set_xlabel("biweekly MA")
axes10[0,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(extreme_oil["oil_ma14"], extreme_oil["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# MA28
sns.regplot(
  ax = axes10[1,0],
  data = extreme_oil,
  x = "oil_ma28",
  y = "sales"
)
axes10[1,0].set_xlabel("monthly MA")
axes10[1,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(extreme_oil["oil_ma28"], extreme_oil["sales"], nan_policy='omit')[0]
      ),
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# MA84
sns.regplot(
  ax = axes10[1,1],
  data = extreme_oil,
  x = "oil_ma84",
  y = "sales"
)
axes10[1,1].set_xlabel("quarterly MA")
axes10[1,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(extreme_oil["oil_ma84"], extreme_oil["sales"], nan_policy='omit')[0]
      ),
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# MA168
sns.regplot(
  ax = axes10[2,0],
  data = extreme_oil,
  x = "oil_ma168",
  y = "sales"
)
axes10[2,0].set_xlabel("semi-annual MA")
axes10[2,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(extreme_oil["oil_ma168"], extreme_oil["sales"], nan_policy='omit')[0]
      ),
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# MA336
sns.regplot(
  ax = axes10[2,1],
  data = extreme_oil,
  x = "oil_ma336",
  y = "sales"
)
axes10[2,1].set_xlabel("annual MA")
axes10[2,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(extreme_oil["oil_ma336"], extreme_oil["sales"], nan_policy='omit')[0]
      ),
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# Show fig10
plt.show()
fig10.savefig("./Plots/LagsEDA/OilMAs.png", dpi=300)
plt.close("all")


# Keep monthly oil MA, filling in NAs
sales_covariates["oil_ma28"] = sales_covariates["oil"].rolling(window = 28, center = False).mean()

# Backwards linear interpolation
sales_covariates["oil_ma28"] = sales_covariates["oil_ma28"].interpolate("linear", limit_direction = "backward")

# Check quality of interpolation
sales_covariates["oil"].plot()
sales_covariates["oil_ma28"].plot()
plt.show()
plt.close("all")


# Cross-correlation of sales & onpromotion
plt.xcorr(sales_covariates["sales"], sales_covariates["onpromotion"], usevlines=True, maxlags=56, normed=True)
plt.grid(True)
plt.ylim([-0.1, 0.1])
plt.xlabel("onpromotion lags / leads")
plt.title("Cross-correlation, decomposed sales\n & differenced onpromotion")
plt.show()
plt.savefig("./Plots/LagsEDA/OnpCCF.png", dpi=300)
plt.close("all")


# FIG11: Onpromotion MAs
fig11, axes11 = plt.subplots(2,2)
fig11.suptitle("Onpromotion change moving averages\n & decomposed sales")

# Calculate MAs without min_periods = 1
onp_ma = sales_covariates.assign(
  onp_ma7 = lambda x: x["onpromotion"].rolling(window = 7, center = False).mean(),
  onp_ma14 = lambda x: x["onpromotion"].rolling(window = 14, center = False).mean(),
  onp_ma28 = lambda x: x["onpromotion"].rolling(window = 28, center = False).mean(),
  onp_ma84 = lambda x: x["onpromotion"].rolling(window = 84, center = False).mean()
)

# MA7
sns.regplot(
  ax = axes11[0,0],
  data = onp_ma,
  x = "onp_ma7",
  y = "sales"
)
axes11[0,0].set_xlabel("weekly MA")
axes11[0,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(onp_ma["onp_ma7"], onp_ma["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# MA14
sns.regplot(
  ax = axes11[0,1],
  data = onp_ma,
  x = "onp_ma14",
  y = "sales"
)
axes11[0,1].set_xlabel("biweekly MA")
axes11[0,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(onp_ma["onp_ma14"], onp_ma["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# MA28
sns.regplot(
  ax = axes11[1,0],
  data = onp_ma,
  x = "onp_ma28",
  y = "sales"
)
axes11[1,0].set_xlabel("monthly MA")
axes11[1,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(onp_ma["onp_ma28"], onp_ma["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# MA84
sns.regplot(
  ax = axes11[1,1],
  data = onp_ma,
  x = "onp_ma84",
  y = "sales"
)
axes11[1,1].set_xlabel("quarterly MA")
axes11[1,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(onp_ma["onp_ma84"], onp_ma["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# Show fig11
plt.show()
fig11.savefig("./Plots/LagsEDA/OnpMAs.png", dpi=300)
plt.close("all")


# FIG12: Onpromotion lags 0 1 6 7
fig12, axes12 = plt.subplots(2,2, sharey=True)
fig12.suptitle("Onpromotion change lags\n & decomposed sales")

# Lag 0
sns.regplot(
  ax = axes12[0,0],
  x = sales_covariates["onpromotion"],
  y = sales_covariates["sales"]
)
axes12[0,0].set_xlabel("lag 0")
axes12[0,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(sales_covariates["onpromotion"], sales_covariates["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )
      
# Lag 1
sns.regplot(
  ax = axes12[0,1],
  x = sales_covariates["onpromotion"].shift(1),
  y = sales_covariates["sales"]
)
axes12[0,1].set_xlabel("lag 1")
axes12[0,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(sales_covariates["onpromotion"].shift(1), sales_covariates["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# Lag 6
sns.regplot(
  ax = axes12[1,0],
  x = sales_covariates["onpromotion"].shift(6),
  y = sales_covariates["sales"]
)
axes12[1,0].set_xlabel("lag 6")
axes12[1,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(sales_covariates["onpromotion"].shift(6), sales_covariates["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# Lag 7
sns.regplot(
  ax = axes12[1,1],
  x = sales_covariates["onpromotion"].shift(7),
  y = sales_covariates["sales"]
)
axes12[1,1].set_xlabel("lag 7")
axes12[1,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(sales_covariates["onpromotion"].shift(7), sales_covariates["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# Show fig12
plt.show()
fig12.savefig("./Plots/LagsEDA/OnpLags.png", dpi=300)
plt.close("all")


# Cross-correlation of sales & transactions
plt.xcorr(sales_covariates["sales"], sales_covariates["transactions"], usevlines=True, maxlags=56, normed=True)
plt.grid(True)
plt.ylim([-0.15, 0.15])
plt.axvline(x = -14, color = "red", linestyle = "dashed")
plt.xlabel("transactions lags / leads")
plt.title("Cross-correlation, decomposed sales\n & differenced transactions")
plt.show()
plt.savefig("./Plots/LagsEDA/TrnsCCF.png", dpi=300)
plt.close("all")


# FIG13: Transactions MAs
fig13, axes13 = plt.subplots(2,2)
fig13.suptitle("Transactions change moving averages\n & decomposed sales")

# Calculate MAs without min_periods = 1
trns_ma = sales_covariates.assign(
  trns_ma7 = lambda x: x["transactions"].rolling(window = 7, center = False).mean(),
  trns_ma14 = lambda x: x["transactions"].rolling(window = 14, center = False).mean(),
  trns_ma28 = lambda x: x["transactions"].rolling(window = 28, center = False).mean(),
  trns_ma84 = lambda x: x["transactions"].rolling(window = 84, center = False).mean()
)

# MA7
sns.regplot(
  ax = axes13[0,0],
  data = trns_ma,
  x = "trns_ma7",
  y = "sales"
)
axes13[0,0].set_xlabel("weekly MA")
axes13[0,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(trns_ma["trns_ma7"], trns_ma["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# MA14
sns.regplot(
  ax = axes13[0,1],
  data = trns_ma,
  x = "trns_ma14",
  y = "sales"
)
axes13[0,1].set_xlabel("biweekly MA")
axes13[0,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(trns_ma["trns_ma14"], trns_ma["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# MA28
sns.regplot(
  ax = axes13[1,0],
  data = trns_ma,
  x = "trns_ma28",
  y = "sales"
)
axes13[1,0].set_xlabel("monthly MA")
axes13[1,0].annotate(
    'Corr={:.2f}'.format(
      spearmanr(trns_ma["trns_ma28"], trns_ma["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# MA84
sns.regplot(
  ax = axes13[1,1],
  data = trns_ma,
  x = "trns_ma84",
  y = "sales"
)
axes13[1,1].set_xlabel("quarterly MA")
axes13[1,1].annotate(
    'Corr={:.2f}'.format(
      spearmanr(trns_ma["trns_ma84"], trns_ma["sales"], nan_policy='omit')[0]
      ), 
      xy=(.6, .9), xycoords="axes fraction", bbox=dict(alpha=0.5)
      )

# Show fig13
plt.show()
fig13.savefig("./Plots/LagsEDA/TrnsMAs.png", dpi=300)
plt.close("all")


# Keep weekly transactions MA, filling in NAs
sales_covariates["trns_ma7"] = sales_covariates["transactions"].rolling(window = 7, center = False).mean()

# Backwards linear interpolation
sales_covariates["trns_ma7"] = sales_covariates["trns_ma7"].interpolate("linear", limit_direction = "backward")

# Check quality of interpolation
sales_covariates["transactions"].plot()
sales_covariates["trns_ma7"].plot()
plt.show()
plt.close("all")


# Save sales_covariates
sales_covariates.to_csv(
  "./ModifiedData/Final/sales_decomped_feats.csv", index=True, encoding="utf-8")

