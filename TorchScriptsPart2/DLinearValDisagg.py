
from darts.metrics import rmse, rmsle, mape, mae, mse


# Define model scoring function for full hierarchy
def scores_hierarchy(val, pred, subset, model, rounding = 4):
  
  def measure_mae(val, pred, subset):
    return mae([val[c] for c in subset], [pred[c] for c in subset])
  
  def measure_mse(val, pred, subset):
    return mse([val[c] for c in subset], [pred[c] for c in subset])
  
  def measure_rmse(val, pred, subset):
    return rmse([val[c] for c in subset], [pred[c] for c in subset])

  def measure_rmsle(val, pred, subset):
    return rmsle([(val[c]) for c in subset], [pred[c] for c in subset])

  scores_dict = {
    "MAE": measure_mae(val, pred, subset),
    "MSE": measure_mse(val, pred, subset),
    "RMSE": measure_rmse(val, pred, subset), 
    "RMSLE": measure_rmsle(val, pred, subset)
      }
      
  print("Model = " + model)    
  
  for key in scores_dict:
    print(
      key + ": mean = " + 
      str(round(np.nanmean(scores_dict[key]), rounding)) + 
      ", sd = " + 
      str(round(np.nanstd(scores_dict[key]), rounding)) + 
      ", min = " + str(round(min(scores_dict[key]), rounding)) + 
      ", max = " + 
      str(round(max(scores_dict[key]), rounding))
       )
       
  print("--------")


# Define function to replace negative predictions with zeroes
def trafo_zeroclip(x):
  return x.map(lambda x: np.clip(x, a_min = 0, a_max = None))


# Load best checkpoint
model_dlinear = DLinear.load_from_checkpoint("DLinearDisagg2.0", best = True)


# All covariates, future & past
dlinear2_futcovars = ['tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'day_sin', 'day_cos', "month_sin", "month_cos", 'oil', 'oil_ma28', 'onpromotion', 'onp_ma28', 'local_holiday', 'regional_holiday', 'national_holiday', 'ny1', 'ny2', 'ny_eve31', 'ny_eve30', 'xmas_before', 'xmas_after', 'quake_after', 'dia_madre', 'futbol', 'black_friday', 'cyber_monday']
dlinear2_pastcovars = ["sales_ema7", "transactions", "trns_ma7"]


# First fit & validate the first series to initialize series
pred_dlinear_disagg = model_dlinear.predict(
  n=15,
  series = y_train_disagg[0],
  future_covariates = x_disagg[0][dlinear2_futcovars],
  past_covariates = x_disagg[0][dlinear2_pastcovars]
  )

# Then loop over all series except first
for i in tqdm(range(1, len(y_train_disagg))):

  # Predict validation data
  pred = model_dlinear.predict(
    n=15,
    series = y_train_disagg[i],
    future_covariates = x_disagg[i][dlinear2_futcovars],
    past_covariates = x_disagg[i][dlinear2_pastcovars]
  )

  # Stack predictions to multivariate series
  pred_dlinear_disagg = pred_dlinear_disagg.stack(pred)

del pred, i


# Score TFT
scores_hierarchy(
  ts_sales[categories_stores][-15:],
  trafo_zeroclip(pred_dlinear_disagg),
  categories_stores,
  "D-Linear (global, all features)"
  )


