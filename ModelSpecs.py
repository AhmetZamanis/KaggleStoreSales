trend & seasonal terms only:

Model=Linear
MAE: mean=2621.4, sd=4923.58, min=5.03, max=19160.13
MSE: mean=70499625.75, sd=212692761.21, min=34.44, max=994111456.04
RMSE: mean=3806.15, sd=7484.17, min=5.87, max=31529.53
RMSLE: mean=0.49, sd=0.25, min=0.3, max=1.2x

+ calendar features (without replacing negative preds with zeroes)

Model=Linear
MAE: mean=2490.91, sd=4729.51, min=4.91, max=18161.36
MSE: mean=53358370.35, sd=157023502.31, min=33.11, max=682423288.06
RMSE: mean=3328.76, sd=6502.13, min=5.75, max=26123.23
RMSLE: mean=0.43, sd=0.23, min=0.2, max=1.28


Model=AutoARIMA (all features)
MAE: mean=3177.22, sd=5938.97, min=3.59, max=24151.07
MSE: mean=65090234.22, sd=174104280.35, min=22.94, max=817507351.93
RMSE: mean=3825.76, sd=7103.09, min=4.79, max=28592.09
RMSLE: mean=0.52, sd=0.61, min=0.2, max=3.67
--------

Model = Linear (time & calendar)
MAE: mean=2490.68, sd=4729.57, min=4.91, max=18161.36
MSE: mean=53357646.8, sd=157023733.27, min=33.11, max=682423288.06
RMSE: mean=3328.56, sd=6502.18, min=5.75, max=26123.23
RMSLE: mean=0.46, sd=0.27, min=0.2, max=1.28

Model = Linear + Random forest (linear decomp, covariate features)
MAE: mean=2949.54, sd=5760.42, min=6.22, max=23345.68
MSE: mean=67404319.58, sd=199819874.33, min=51.01, max=927516253.58
RMSE: mean=3729.04, sd=7314.27, min=7.14, max=30455.15
RMSLE: mean=0.5, sd=0.4, min=0.16, max=1.9
--------

Model = Linear + XGBoost untuned (linear decomp, covariate features)
MAE: mean=3060.46, sd=5984.15, min=6.35, max=24577.0
MSE: mean=76354996.81, sd=223252746.14, min=54.88, max=1061127656.54
RMSE: mean=3989.13, sd=7774.43, min=7.41, max=32574.95
RMSLE: mean=0.53, sd=0.47, min=0.19, max=1.89
--------

Model = Linear (time only)
MAE: mean = 2621.4, sd = 4923.58, min = 5.03, max = 19160.13
MSE: mean = 70499625.75, sd = 212692761.21, min = 34.44, max = 994111456.04
RMSE: mean = 3806.15, sd = 7484.17, min = 5.87, max = 31529.53
RMSLE: mean = 0.49, sd = 0.25, min = 0.3, max = 1.26
--------
Model = Linear + Random forest (STL decomp, covariate + calendar features)
MAE: mean = 2775.22, sd = 5294.9, min = 5.11, max = 22540.5
MSE: mean = 68710801.31, sd = 212215204.9, min = 35.71, max = 1044254758.16
RMSE: mean = 3741.69, sd = 7396.66, min = 5.98, max = 32314.93
RMSLE: mean = 0.48, sd = 0.27, min = 0.26, max = 1.3
--------
Model = Linear + XGBoost untuned (STL decomp, covariate + calendar features)
MAE: mean = 2986.53, sd = 5742.79, min = 5.15, max = 25868.4
MSE: mean = 79903065.08, sd = 259141086.04, min = 36.09, max = 1394927088.03
RMSE: mean = 4095.4, sd = 7945.49, min = 6.01, max = 37348.72
RMSLE: mean = 0.51, sd = 0.36, min = 0.26, max = 1.82
--------

Model = Linear (time & calendar)
MAE: mean=2490.68, sd=4729.57, min=4.91, max=18161.36
MSE: mean=53357646.8, sd=157023733.27, min=33.11, max=682423288.06
RMSE: mean=3328.56, sd=6502.18, min=5.75, max=26123.23
RMSLE: mean=0.46, sd=0.27, min=0.2, max=1.28
--------
Model = Linear + Random forest (STL decomp, covariate features)
MAE: mean = 2689.02, sd = 5158.25, min = 4.97, max = 21686.3
MSE: mean = 58245076.48, sd = 176222421.86, min = 33.96, max = 849969897.75
RMSE: mean = 3483.65, sd = 6790.38, min = 5.83, max = 29154.24
RMSLE: mean = 0.44, sd = 0.28, min = 0.15, max = 1.31
--------
Model = Linear + XGBoost untuned (STL decomp, covariate features)
MAE: mean = 3031.09, sd = 5880.44, min = 4.92, max = 25721.48
MSE: mean = 78410131.67, sd = 243713513.61, min = 33.87, max = 1260756454.48
RMSE: mean = 4075.08, sd = 7861.54, min = 5.82, max = 35507.13
RMSLE: mean = 0.53, sd = 0.48, min = 0.19, max = 2.69
--------







# LINEAR




# XGBOOST

# Category global

# Fit XGB model
model_xgb_cat.fit(
  [decomp_cat["AUTOMOTIVE":"SEAFOOD"][category] for category in categories],
  future_covariates = [x[xgb_covars] for x in x_cat]
)

# Predict validation data for each category
pred_xgb_cat_list = model_xgb_cat.predict(
  n = 227,
  series = [decomp_cat["AUTOMOTIVE":"SEAFOOD"][category] for category in categories],
  future_covariates = [x[xgb_covars] for x in x_cat]
  )

# Stack predictions to get multivariate series
pred_xgb_cat = pred_xgb_cat_list[0].stack(pred_xgb_cat_list[1])
for pred in pred_xgb_cat_list[2:]:
  pred_xgb_cat = pred_xgb_cat.stack(pred)






# D-LINEAR

# 1
model_dlinear_cat = DLinear(
  input_chunk_length = 30,
  output_chunk_length = 1,
  kernel_size = 25,
  batch_size = 32,
  n_epochs = 500,
  model_name = "DLinearCat1",
  log_tensorboard = True,
  save_checkpoints = True,
  random_state = 1923,
  pl_trainer_kwargs = {
    "callbacks": [early_stopper, progress_bar],
    "accelerator": "gpu",
    "devices": [0]
    },
  show_warnings = True,
  force_reset = True
)



# RNN

# 1
model_rnn_cat = RNN(
  model = "LSTM",
  input_chunk_length = 30,
  training_length = 45,
  batch_size = 32,
  n_epochs = 500,
  n_rnn_layers = 2,
  hidden_dim = 32,
  dropout = 0.2,
  model_name = "RNNCat1",
  log_tensorboard = True,
  save_checkpoints = True,
  random_state = 1923,
  pl_trainer_kwargs = {
    "callbacks": [early_stopper, progress_bar],
    "accelerator": "gpu",
    "devices": [0]
    },
  show_warnings = True,
  force_reset = True
)
