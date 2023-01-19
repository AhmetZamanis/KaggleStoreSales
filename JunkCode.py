# df_agg = df_train.groupby("date").agg(
#   {
#     "sales": "mean",
#     "onpromotion": "sum",
#     "transactions": "sum",
#     "oil": "mean",
#     "local_holiday": "mean",
#     "regional_holiday": "mean",
#     "national_holiday": "mean",
#     "event": "mean"
#   }
# )







# Plot annual seasonality, quarters aggregated
sales_quarterly = total_sales[(total_sales.index.year < 2017) | (total_sales.index.month < 7)]
sales_quarterly = sales_quarterly.groupby([(sales_quarterly.index.quarter), (sales_quarterly.index.year)]).sum()
sales_quarterly.index.names = "quarter", "year"
sales_quarterly = sales_quarterly.reset_index()
sales_quarterly.sales = sales_quarterly.sales / 1000000

sns.lineplot(
  x = sales_quarterly.quarter.astype(str),
  y = sales_quarterly.sales,
  hue = sales_quarterly.year.astype(str),
  data = sales_quarterly
)
plt.ylabel("quarterly sales, millions")
plt.xlabel("quarter")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='upper left')
plt.show()
plt.close("all")


# Plot annual seasonality, months aggregated
sales_monthly = total_sales[(total_sales.index.year < 2017) | (total_sales.index.month < 8)]
sales_monthly = sales_monthly.groupby([(sales_monthly.index.month), (sales_monthly.index.year)]).sum()
sales_monthly.index.names = "month", "year"
sales_monthly = sales_monthly.reset_index()
sales_monthly.sales = sales_monthly.sales / 1000000

sns.lineplot(
  x = sales_monthly.month.astype(str),
  y = sales_monthly.sales,
  hue = sales_monthly.year.astype(str),
  data = sales_monthly
)
plt.ylabel("monthly sales, millions")
plt.xlabel("month")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='upper left')
plt.show()
plt.close("all")


# Plot annual seasonality, weeks aggregated
sales_weekly = total_sales[
  (total_sales.index.year < 2017) | ((total_sales.index.month < 8) & (~total_sales.index.week.isin([31,52])))]
sales_weekly = sales_weekly.groupby([(sales_weekly.index.week), (sales_weekly.index.year)]).sum()
sales_weekly.index.names = "week", "year"
sales_weekly = sales_weekly.reset_index()
sales_weekly.sales = sales_weekly.sales / 1000000

sns.lineplot(
  x = sales_weekly.week,
  y = sales_weekly.sales,
  hue = sales_weekly.year.astype(str),
  data = sales_weekly
)
plt.ylabel("weekly sales, millions")
plt.xlabel("week")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='upper left')
plt.show()
plt.close("all")


# Plot annual seasonality, day of year aggregated
sales_dayofyear =  total_sales[(total_sales.index.year < 2017) | (total_sales.index.month < 8)]
sales_dayofyear= sales_dayofyear.groupby([(sales_dayofyear.index.dayofyear), (sales_dayofyear.index.year)]).sum()
sales_dayofyear.index.names = "dayofyear", "year"
sales_dayofyear = sales_dayofyear.reset_index()
sales_dayofyear.sales = sales_dayofyear.sales / 1000000

sns.lineplot(
  x = sales_dayofyear.dayofyear,
  y = sales_dayofyear.sales,
  hue = sales_dayofyear.year.astype(str),
  data = sales_dayofyear
)
plt.ylabel("daily sales, millions")
plt.xlabel("day of year")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='upper left')
plt.show()
plt.close("all")



# Plot monthly seasonality, days of month aggregated
sales_dayofmonth = total_sales[(total_sales.index.year < 2017)]
sales_dayofmonth = sales_dayofmonth.groupby([(sales_dayofmonth.index.day), (sales_dayofmonth.index.year)]).sum()
sales_dayofmonth.index.names = "day", "year"
sales_dayofmonth = sales_dayofmonth.reset_index()
sales_dayofmonth.sales = sales_dayofmonth.sales / 1000000

sns.lineplot(
  x = sales_dayofmonth.day,
  y = sales_dayofmonth.sales,
  hue = sales_dayofmonth.year.astype(str),
  data = sales_dayofmonth
)
plt.ylabel("sales, millions")
plt.xlabel("day of month")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='upper left')
plt.show()
plt.close("all")


# Plot weekly seasonality, days of week aggregated
sales_dayofweek = total_sales[(total_sales.index.year < 2017) | (total_sales.index.month < 8)]
sales_dayofweek = sales_dayofweek.groupby([(sales_dayofweek.index.dayofweek), (sales_dayofweek.index.year)]).sum()
sales_dayofweek.index.names = "day", "year"
sales_dayofweek = sales_dayofweek.reset_index()
sales_dayofweek.sales = sales_dayofweek.sales / 1000000

sns.lineplot(
  x = (sales_dayofweek.day + 1).astype(str),
  y = sales_dayofweek.sales,
  hue = sales_dayofweek.year.astype(str),
  data = sales_dayofweek
)
plt.ylabel("sales, millions")
plt.xlabel("day of week")
plt.legend(title = "year", bbox_to_anchor=(1.05, 1.0), fontsize="small", loc='upper left')
plt.show()
plt.close("all")



from sktime.utils.plotting import plot_correlations
plot_correlations(total_sales)


# Lag plots
from sktime.utils.plotting import plot_lags
fig3, ax3 = plot_lags(total_sales, lags=[1,2,3,4,5,6,7])
plt.show()
plt.close("all")



from darts.utils.statistics import plot_acf
from darts.utils.statistics import plot_pacf
from darts.utils.statistics import plot_residuals_analysis
fig3, axes3 = plt.subplots(2)
fig3.suptitle("ACF and PACF plots, daily sales")

# ACF plot
plot_acf(ts_total, axis=axes3[0], max_lag=54, bartlett_confint=False)

# Show fig3
plt.show()
fig3.savefig("./", dpi=300)
plt.close("all")

plot_acf(ts_total["sales"], max_lag=54)
plot_pacf(ts_total["sales"], max_lag=54)
plot_residuals_analysis(ts_total["sales"])


# STL decomposition
from statsmodels.tsa.seasonal import STL
stl_monthly = STL(np.log(total_sales), period=28, robust=True).fit()
stl_monthly.plot()
stl_monthly.trend




df["xmas_before"] = 0
df.loc[(df.index.day == 23) & (df.index.month == 12), "xmas_before"] = 11
df.loc[(df.index.day.isin([21,22])) & (df.index.month == 12), "xmas_before"] = 10
df.loc[(df.index.day == 20) & (df.index.month == 12), "xmas_before"] = 9
df.loc[(df.index.day.isin([18,19])) & (df.index.month == 12), "xmas_before"] = 8
df.loc[(df.index.day == 17) & (df.index.month == 12), "xmas_before"] = 7
df.loc[(df.index.day == 16) & (df.index.month == 12), "xmas_before"] = 6
df.loc[(df.index.day == 15) & (df.index.month == 12), "xmas_before"] = 5
df.loc[(df.index.day == 14) & (df.index.month == 12), "xmas_before"] = 4
df.loc[(df.index.day == 13) & (df.index.month == 12), "xmas_before"] = 3

df["xmas_after"] = 0
df.loc[(df.index.day == 23) & (df.index.month == 12), "xmas_before"] = 5
df.loc[(df.index.day == 24) & (df.index.month == 12), "xmas_before"] = 4
df.loc[(df.index.day == 25) & (df.index.month == 12), "xmas_before"] = 3
df.loc[(df.index.day == 26) & (df.index.month == 12), "xmas_before"] = 2
df.loc[(df.index.day == 27) & (df.index.month == 12), "xmas_before"] = 1


from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.utils.utils import ModelMode
from darts.utils.utils import SeasonalityMode


model_exp = ExponentialSmoothing(
  trend = ModelMode.ADDITIVE,
  seasonal = SeasonalityMode.ADDITIVE,
  seasonal_periods = 7)
  

model_exp.fit(y_train)
pred_exp = model_exp.predict(n = 227)


from statsmodels.tsa.stattools import ccf
ccfs = pd.DataFrame(
  {
    "oil": ccf(sales_covariates["oil"], sales_covariates["sales"]),
    "onpromotion": ccf(sales_covariates["onpromotion"], sales_covariates["sales"]),
    "transactions": ccf(sales_covariates["transactions"], sales_covariates["sales"]),
  }
)


# Calculate cross-correlations of sales and covariates
ccfs = pd.DataFrame.from_dict(
    {x: [sales_covariates["sales"].corr(sales_covariates[x].shift(t)) for t in range(0,181)] for x in sales_covariates.columns})


# FIG10: Oil vs sales timeplots
fig10, axes10 = plt.subplots(2, sharex=True)
fig10.suptitle("Oil and sales")

# Sales
sns.lineplot(
  ax = axes10[0],
  x = sales_covariates.index,
  y = "sales",
  data = sales_covariates
)
axes10[0].set_ylabel("sales, decomposed")

# Oil
sns.lineplot(
  ax = axes10[1],
  x = sales_covariates.index,
  y = "oil",
  data = sales_covariates
)
axes10[1].set_ylabel("oil, differenced")

# Show fig10
plt.show()
fig10.savefig("./Plots/LagsEDA/OilTime.png", dpi=300)
plt.close("all")


# Cross correlation
sns.barplot(
  x = -ccfs.index,
  y = ccfs.oil
)
plt.title("Correlation of sales & oil lags")
plt.xlabel("lags")
plt.ylabel("correlation")

plt.show()
plt.close("all")


from statsmodels.tsa.stattools import grangercausalitytests as granger
granger_oil = granger(sales_covariates[["sales", "oil"]], maxlag=180)



# FIG10: Regplots of oil moving averages & sales
fig10, axes10 = plt.subplots(2,2, sharey=True)
fig10.suptitle("Oil price change moving averages\n & decomposed sales")

# MA7
sns.regplot(
  ax = axes10[0,0],
  data = sales_covariates,
  x = "oil_ma7",
  y = "sales"
)
axes10[0,0].set_xlabel("weekly MA")

# MA14
sns.regplot(
  ax = axes10[0,1],
  data = sales_covariates,
  x = "oil_ma14",
  y = "sales"
)
axes10[0,1].set_xlabel("biweekly MA")

# MA28
sns.regplot(
  ax = axes10[1,0],
  data = sales_covariates,
  x = "oil_ma28",
  y = "sales"
)
axes10[1,0].set_xlabel("monthly MA")

# MA84
sns.regplot(
  ax = axes10[1,1],
  data = sales_covariates,
  x = "oil_ma84",
  y = "sales"
)
axes10[1,1].set_xlabel("quarterly MA")


rng = np.random.default_rng(1923)
for column in extreme_oil.columns:
  column_mean = extreme_oil[column].mean()
  column_sd = extreme_oil[column].std()
  na_filler = pd.Series(rng.normal(loc=column_mean, scale=column_sd, size=len(extreme_oil[column])))
  extreme_oil[column] = extreme_oil[column].fillna(na_filler)


# Random distribution interpolation
rng = np.random.default_rng(1923)
mu = sales_covariates["oil_ma28"].mean()
sd = sales_covariates["oil_ma28"].std()
na_filler = pd.Series(rng.normal(loc=mu, scale=sd, size=len(sales_covariates["oil_ma28"])))
sales_covariates["oil_ma28"] = sales_covariates["oil_ma28"].fillna(na_filler)



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

sales_covariates = sales_covariates.drop([
  "oil_ma7", "oil_ma14", "oil_ma28", "oil_ma84", "oil_ma168", "oil_ma336"], axis = 1)
  
  
ccfs = pd.DataFrame.from_dict(
    {x: [sales_covariates["sales"].corr(sales_covariates[x].shift(t)) for t in range(-14,15)] for x in sales_covariates.columns})


sns.barplot(
  x = ccfs.index,
  y = ccfs.onpromotion
)
plt.title("Correlation of sales & onpromotion lags")
plt.xlabel("lags")
plt.ylabel("correlation")

plt.show()
plt.close("all")



# Cross-correlation of oil and sales
plt.xcorr(sales_covariates["sales"], sales_covariates["oil"], usevlines=True, maxlags=336, normed=True, lw=2)
plt.grid(True)
plt.ylim([-0.3, 0.3])
plt.xlabel("oil lags / leads")
plt.title("Cross-correlation, decomposed sales & oil price change")
plt.show()
plt.close("all")


# Retrieve predictions and residuals for 2014-2017
res_linear = model_linear.residuals(
  trafo_log(ts), future_covariates = ts_timefeats, forecast_horizon = 1,
  verbose = True)
sales_decomped = res_linear[350:]
preds_time = trafo_log(ts[365:]) - res_linear[350:]


# Save 17 predictions of model 1 to be added later to model 2 17 predictions
pred_linear.pd_dataframe().to_csv(
  "./ModifiedData/Final/preds_model1_17.csv", index=True, encoding="utf-8")
  
  
# Load 17 time preds
preds_time17 = TimeSeries.from_csv(
  "./ModifiedData/Final/preds_model1_17.csv", time_col = "date", freq = "D")


sales_covariates["oil_ma336"] = sales_covariates["oil_ma336"].interpolate("spline", order = 3, limit_direction = "backward")
