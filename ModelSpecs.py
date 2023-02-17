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
