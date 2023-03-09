

# Initialize list of store covariates
store_covars = []

for store in [int(store) for store in stores]:
  
  # Retrieve common covariates
  covars = common_covars.copy()
  
  # Retrieve local & regional holiday
  covars["local_holiday"] = df[
    df["store_nbr"] == store].groupby("date").local_holiday.mean()
    
  covars["regional_holiday"] = df[
    df["store_nbr"] == store].groupby("date").regional_holiday.mean()  
  
  # Retrieve differenced sales EMA
  covars["sales_ema7"] = diff.fit_transform(
    df[df["store_nbr"] == store].groupby("date").sales.sum()
    ).interpolate(
  "linear", limit_direction = "backward"
  ).rolling(
    window = 7, min_periods = 1, center = False, win_type = "exponential").mean()
    
  # Retrieve differenced onpromotion, its MA
  covars["onpromotion"] = diff.fit_transform(
    df[df["store_nbr"] == store].groupby("date").onpromotion.sum()
    ).interpolate(
      "time", limit_direction = "both"
      )
      
  covars["onp_ma28"] = covars["onpromotion"].rolling(
    window = 28, center = False
    ).mean().interpolate(
  method = "spline", order = 2, limit_direction = "both"
  ) 
  
  # Retrieve differenced transactions, its MA
  covars["transactions"] = diff.fit_transform(
    df[df["store_nbr"] == store].groupby("date").transactions.sum().interpolate(
      "time", limit_direction = "both"
      )
    )
    
  covars["trns_ma7"] = covars["transactions"].rolling(
    window = 7, center = False
    ).mean().interpolate(
  "linear", limit_direction = "backward"
  )
  
  # Create darts TS, fill gaps
  covars = na_filler.transform(
    TimeSeries.from_dataframe(covars, freq = "D")
    )
  
  # Cyclical encode day of month using datetime_attribute_timeseries
  covars = covars.stack(
    datetime_attribute_timeseries(
      time_index = covars,
      attribute = "day",
      cyclic = True
      )
    )
    
   # Cyclical encode month using datetime_attribute_timeseries
  covars = covars.stack(
    datetime_attribute_timeseries(
      time_index = covars,
      attribute = "month",
      cyclic = True
      )
    )
    
  # Append TS to list
  store_covars.append(covars)
  
# Cleanup
del covars, store


# Create dataframe where column=static covariate and index=store nbr
store_static = df[["store_nbr", "city", "state", "store_type", "store_cluster"]].reset_index().drop("date", axis=1).drop_duplicates().set_index("store_nbr")
store_static["store_cluster"] = store_static["store_cluster"].astype(str)

# Encode static covariates
store_static = pd.get_dummies(store_static, sparse = False, drop_first = True)


# Create min-max scaler
scaler_minmax = Scaler()

# Train-validation split and scaling for covariates
x_store = []
for series in store_covars:
  
  # Split train-val series
  cov_train, cov_innerval, cov_outerval = series[:-76], series[-76:-31], series[-31:]
  
  # Scale train-val series
  cov_train = scaler_minmax.fit_transform(cov_train)
  cov_innerval = scaler_minmax.transform(cov_innerval)
  cov_outerval = scaler_minmax.transform(cov_outerval)
  
  # Rejoin series
  cov_train = (cov_train.append(cov_innerval)).append(cov_outerval)
  
  # Cast series to 32-bits for performance gains
  cov_train = cov_train.astype(np.float32)
  
  # Append series to list
  x_store.append(cov_train)
  
# Cleanup
del cov_train, cov_innerval, cov_outerval


# List of store sales
store_sales = [ts_sales[store] for store in stores]

# Train-validation split for store sales
y_train_store, y_val_store = [], []
for series in store_sales:
  
  # Add static covariates to series
  series = series.with_static_covariates(
    store_static[store_static.index == int(series.components[0])]
  )
  
  # Split train-val series
  y_train, y_val = series[:-15], series[-15:]
  
  # Cast series to 32-bits for performance gains
  y_train = y_train.astype(np.float32)
  y_val = y_val.astype(np.float32)
  
  # Append series
  y_train_store.append(y_train)
  y_val_store.append(y_val)
  
# Cleanup
del y_train, y_val
