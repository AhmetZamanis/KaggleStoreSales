
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar

# Create early stopper
early_stopper = EarlyStopping(
  monitor = "val_loss",
  min_delta = 5000, # 1% of min. MSE of best model so far
  patience = 10
)

# Progress bar
progress_bar = RichProgressBar()

# # Rich model summary
# model_summary = RichModelSummary(max_depth = -1)


# # Specify TFT model 2.0 (TFT specific params all default)
# model_tft = TFTModel(
#   input_chunk_length = 30,
#   output_chunk_length = 15,
#   hidden_size = 16,
#   lstm_layers = 1,
#   num_attention_heads = 4,
#   dropout = 0.1,
#   hidden_continuous_size = 8,
#   batch_size = 32,
#   n_epochs = 500,
#   likelihood = None,
#   loss_fn = torch.nn.MSELoss(),
#   model_name = "TFTStoreX",
#   log_tensorboard = True,
#   save_checkpoints = True,
#   show_warnings = True,
#   optimizer_kwargs = {"lr": 0.002},
#   lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
#   lr_scheduler_kwargs = {"patience": 5},
#   pl_trainer_kwargs = {
#     "callbacks": [early_stopper],
#     "accelerator": "gpu",
#     "devices": [0]
#     }
# )


# # Specify TFT model 2.1 (TFT specific params all default, local-regional fix,
# # higher initial LR)
# model_tft = TFTModel(
#   input_chunk_length = 30,
#   output_chunk_length = 15,
#   hidden_size = 16,
#   lstm_layers = 1,
#   num_attention_heads = 4,
#   dropout = 0.1,
#   hidden_continuous_size = 8,
#   batch_size = 32,
#   n_epochs = 500,
#   likelihood = None,
#   loss_fn = torch.nn.MSELoss(),
#   model_name = "TFTStore2.1",
#   log_tensorboard = True,
#   save_checkpoints = True,
#   show_warnings = True,
#   optimizer_kwargs = {"lr": 0.005},
#   lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
#   lr_scheduler_kwargs = {"patience": 5},
#   pl_trainer_kwargs = {
#     "callbacks": [early_stopper],
#     "accelerator": "gpu",
#     "devices": [0]
#     }
# )


# # Specify TFT model 2.2 (TFT specific params all default, local-regional fix,
# # initial LR 0.003)
# model_tft = TFTModel(
#   input_chunk_length = 30,
#   output_chunk_length = 15,
#   hidden_size = 16,
#   lstm_layers = 1,
#   num_attention_heads = 4,
#   dropout = 0.1,
#   hidden_continuous_size = 8,
#   batch_size = 32,
#   n_epochs = 500,
#   likelihood = None,
#   loss_fn = torch.nn.MSELoss(),
#   model_name = "TFTStore2.2",
#   log_tensorboard = True,
#   save_checkpoints = True,
#   show_warnings = True,
#   optimizer_kwargs = {"lr": 0.003},
#   lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
#   lr_scheduler_kwargs = {"patience": 5},
#   pl_trainer_kwargs = {
#     "callbacks": [early_stopper],
#     "accelerator": "gpu",
#     "devices": [0]
#     }
# )


# Specify TFT model 2.3 (TFT specific params all default, local-regional fix,
# # initial LR 0.002)
model_tft = TFTModel(
  input_chunk_length = 30,
  output_chunk_length = 15,
  hidden_size = 16,
  lstm_layers = 1,
  num_attention_heads = 4,
  dropout = 0.1,
  hidden_continuous_size = 8,
  batch_size = 32,
  n_epochs = 500,
  likelihood = None,
  loss_fn = torch.nn.MSELoss(),
  model_name = "TFTStore2.3",
  log_tensorboard = True,
  save_checkpoints = True,
  show_warnings = True,
  optimizer_kwargs = {"lr": 0.002},
  lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
  lr_scheduler_kwargs = {"patience": 5, "verbose": True},
  pl_trainer_kwargs = {
    "callbacks": [early_stopper],
    "accelerator": "gpu",
    "devices": [0]
    }
)


# All covariates, future & past
tft_futcovars = [
  "trend", 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 
  'day_sin', 'day_cos', "month_sin", "month_cos", 'oil', 'oil_ma28', 'onpromotion', 
  'onp_ma28', 'local_holiday', 'regional_holiday', 'national_holiday', 'ny1', 
  'ny2', 'ny_eve31', 'ny_eve30', 'xmas_before', 'xmas_after', 'quake_after', 
  'dia_madre', 'futbol', 'black_friday', 'cyber_monday']

tft_pastcovars = ["sales_ema7", "transactions", "trns_ma7"]


# Fit TFT model
model_tft.fit(
  series = [y[:-45] for y in y_train_store],
  future_covariates = [x[tft_futcovars] for x in x_store],
  past_covariates = [x[tft_pastcovars] for x in x_store],
  val_series = [y[-45:] for y in y_train_store],
  val_future_covariates = [x[tft_futcovars] for x in x_store],
  val_past_covariates = [x[tft_pastcovars] for x in x_store],
  verbose = True,
  num_loader_workers = 6
)





