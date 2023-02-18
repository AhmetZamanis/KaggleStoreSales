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
