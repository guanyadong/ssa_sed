import logging
import math
import os
import pandas as pd

workspace = ".."
# DESED Paths
weak = os.path.join(workspace, 'dataset/metadata/train/weak.tsv')
unlabel = os.path.join(workspace, 'dataset/metadata/train/unlabel_in_domain.tsv')
synthetic = os.path.join(workspace, 'dataset/metadata/train/synthetic20/soundscapes.tsv')
validation = os.path.join(workspace, 'dataset/metadata/validation/validation.tsv')
test2018 = os.path.join(workspace, 'dataset/metadata/validation/test_dcase2018.tsv')
eval2018 = os.path.join(workspace, 'dataset/metadata/validation/eval_dcase2018.tsv')
eval_desed = os.path.join(workspace, "dataset/metadata/eval/public.tsv")
# Useful because does not correspond to the tsv file path (metadata replace by audio), (due to subsets test/eval2018)
audio_validation_dir = os.path.join(workspace, 'dataset/audio/validation')
audio_eva_dir = os.path.join(workspace,'/dataset/audio/eval/public')

# Scaling data
scaler_type = "dataset"

# Data preparation
ref_db = -55
sample_rate = 16000
max_len_seconds = 10.
# features
n_window = 2048
hop_size = 255
n_mels = 128
max_frames = math.ceil(max_len_seconds * sample_rate / hop_size)
mel_f_min = 0.
mel_f_max = 8000.

# Model
max_consistency_cost = 2

# Training
in_memory = True
in_memory_unlab = False
num_workers = 8
batch_size = 24
batch_size_mixup = 36
noise_snr = [100, 120]

n_epoch = 200
n_epoch_rampup = 80
n_epoch_rampdown = 250
checkpoint_epochs = 1
save_best = True
early_stopping = None
es_init_wait = 50  # es for early stopping
adjust_lr = True
max_learning_rate = 0.001  # Used if adjust_lr is True
default_learning_rate = 0.001  # Used if adjust_lr is False

# Post processing
median_window_s = 0.45

median_window_short = 0.3
median_window_media = 0.9
median_window_long = 2.5


# Classes
file_path = os.path.abspath(os.path.dirname(__file__))
classes = pd.read_csv(os.path.join(file_path, validation), sep="\t").event_label.dropna().sort_values().unique()
# Logger
terminal_level = logging.INFO
