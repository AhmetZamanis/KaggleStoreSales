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
sales_covariates.assign(
  oil_ema7
)

= sales_covariates["sales"].rolling(
  window = 9, min_periods = 1, center = False, win_type = "exponential").mean()


