import os

import pandas as pd

from matplotlib import pyplot as plt

results_csv_file_name = r"results_cnn_configurations_all_workbook.csv"
data_dir = r"D:\Documents\Academics\BME517\final_project\data"
output_dir = r"D:\Documents\Academics\BME517\final_project\output"
figsize = (12, 12)
ylim = (-0.01, 0.11)

df_results = pd.read_csv(os.path.join(data_dir, results_csv_file_name))

fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="all", sharey="all")
for (conv_mode, fc_mode), df_group in df_results.groupby(["conv", "fc"]):
    conv_idx = 0 if conv_mode == "baseline" else 1
    fc_idx = 0 if fc_mode == "baseline" else 1
    for pool_mode, df_trial in df_group.groupby("pool"):
        n_params = df_trial["n_params"].values
        mean_val_error = 1 - df_trial["mean_val_acc"].values
        axes[conv_idx][fc_idx].plot(n_params, mean_val_error, marker="o", label=pool_mode)
    axes[conv_idx][fc_idx].set_title(f"n_params: conv_mode={conv_mode}, fc_mode={fc_mode}")
    axes[conv_idx][fc_idx].legend()
    axes[conv_idx][fc_idx].set_xlabel("n_params")
    axes[conv_idx][fc_idx].set_ylabel("error rate")
    axes[conv_idx][fc_idx].set_xscale("log")
    # axes[conv_idx][fc_idx].set_yscale("log")
    axes[conv_idx][fc_idx].set_ylim(ylim)
plt.tight_layout()
plt.show()
plt.close()

fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="all", sharey="all")
for (conv_mode, fc_mode), df_group in df_results.groupby(["conv", "fc"]):
    conv_idx = 0 if conv_mode == "baseline" else 1
    fc_idx = 0 if fc_mode == "baseline" else 1
    for pool_mode, df_trial in df_group.groupby("pool"):
        n_mult = df_trial["n_mult"].values
        mean_val_error = 1 - df_trial["mean_val_acc"].values
        axes[conv_idx][fc_idx].plot(n_mult, mean_val_error, marker="o", label=pool_mode)
    axes[conv_idx][fc_idx].set_title(f"n_mult: conv_mode={conv_mode}, fc_mode={fc_mode}")
    axes[conv_idx][fc_idx].legend()
    axes[conv_idx][fc_idx].set_xlabel("n_mult")
    axes[conv_idx][fc_idx].set_ylabel("error rate")
    axes[conv_idx][fc_idx].set_xscale("log")
    # axes[conv_idx][fc_idx].set_yscale("log")
    axes[conv_idx][fc_idx].set_ylim(ylim)

plt.tight_layout()
plt.show()
plt.close()




