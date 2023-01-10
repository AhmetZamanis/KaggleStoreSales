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


