import os
import warnings
# os.environ["OMP_NUM_THREADS"] = '7'  # avoid memory leak in sklearn

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as k

from compute_acc import compute_acc
from construct_cnn1d import construct_cnn1d
from PCAGMM import PCAGMM
from WaveClusDataset import WaveClusDataset


# The k-means implementation in Scikit-Learn has a (reportedly) miniscule memory leak.
# As the leak is not problematic, this suppresses the warning regarding this leak.
# Alternatively, uncomment the " os.environ["OMP_NUM_THREADS"] = '7' " line above to avoid the leak.
warnings.filterwarnings(action="ignore", message="KMeans is known to have a memory leak.*")


print(tf.config.list_physical_devices('GPU'))


def load_wave_clus_datasets(dir_wave_clus, verbose: bool = False):
    diff_noise_combos = [
        ("Easy1", "005"), ("Easy1", "01"), ("Easy1", "015"), ("Easy1", "02"),
        ("Easy1", "025"), ("Easy1", "03"), ("Easy1", "035"), ("Easy1", "04"),
        ("Easy2", "005"), ("Easy2", "01"), ("Easy2", "015"), ("Easy2", "02"),
        ("Difficult1", "005"), ("Difficult1", "01"), ("Difficult1", "015"), ("Difficult1", "02"),
        ("Difficult2", "005"), ("Difficult2", "01"), ("Difficult2", "015"), ("Difficult2", "02")]
    datasets = dict()
    for diff_str, noise_str in diff_noise_combos:
        noise_float = float(f"0.{noise_str[1:]}")
        dataset_filename = f"C_{diff_str}_noise{noise_str}.mat"
        dataset_filepath = os.path.join(dir_wave_clus, dataset_filename)
        print(f"[INFO] Loading dataset with difficulty {diff_str}, noise {noise_float}:") if verbose else None
        dataset = WaveClusDataset()
        dataset.load(dataset_filepath)
        datasets[(diff_str, noise_float)] = dataset
    return datasets


def construct_and_count_one_configuration(
        conv_sizes: list[int], pool_sizes: list[int], fc_sizes: list[int]):

    n_inputs = 64
    n_kernel = 3
    n_outputs = 3

    float_multiply_count = 0

    convolutional_layers = []
    curr_length = n_inputs
    prev_conv_size = 1
    for pool_size, conv_size in zip(pool_sizes[:-1], conv_sizes):
        if pool_size != 0:
            convolutional_layers.append(("pool", pool_size))
            curr_length /= pool_size
        convolutional_layers.append(("conv", conv_size))
        float_multiply_count += curr_length * conv_size * prev_conv_size * n_kernel
        prev_conv_size = conv_size

    if pool_sizes[-1] != 0:
        convolutional_layers.append(("pool", pool_sizes[-1]))
        curr_length /= pool_sizes[-1]

    prev_fc_size = curr_length * prev_conv_size
    for fc_size in fc_sizes:
        float_multiply_count += fc_size * prev_fc_size
        prev_fc_size = fc_size

    float_multiply_count += n_outputs * prev_fc_size

    return {"convolutional_layers": convolutional_layers, "fully_connected_layers": fc_sizes}, float_multiply_count


def construct_and_count_all_configurations():

    conv_sizes_baseline = [32, 64, 128, 128]
    conv_sizes_reduced = [
        [16, 32, 64, 64],
        [8, 16, 32, 32],
        [4, 8, 16, 16],
        [2, 4, 8, 8],
        [1, 2, 4, 4]]
    fc_sizes_baseline = [300, 100]
    fc_sizes_reduced = [
        [150, 50],
        [75, 25],
        [36, 12],
        [18, 6],
        [9, 3]]
    pool_sizes_baseline = [0, 0, 2, 2, 0]
    pool_sizes_frequent = [
        [0, 0, 2, 2, 2],
        [0, 2, 2, 2, 0],
        [0, 2, 2, 2, 2],
        [2, 0, 2, 2, 0],
        [2, 0, 2, 2, 2],
        [2, 2, 2, 2, 0],
        [2, 2, 2, 2, 2]]
    pool_sizes_aggressive = [
        [0, 0, 2, 4, 0],
        [0, 0, 4, 2, 0],
        [0, 0, 4, 4, 0],
        [0, 0, 4, 8, 0],
        [0, 0, 8, 4, 0],
        [0, 0, 8, 8, 0]]

    configurations_and_counts = [
        construct_and_count_one_configuration(
            conv_sizes=conv_sizes_baseline, pool_sizes=pool_sizes_baseline, fc_sizes=fc_sizes_baseline)]
    for conv_sizes_mode in ["reduced", "baseline"]:
        for fc_sizes_mode in ["reduced", "baseline"]:
            for pool_sizes_mode in ["frequent", "aggressive", "baseline"]:

                if conv_sizes_mode == "baseline" and fc_sizes_mode == "baseline" and pool_sizes_mode == "baseline":
                    continue

                n_steps = 1000
                if conv_sizes_mode == "reduced":
                    n_steps = min(n_steps, len(conv_sizes_reduced))
                if fc_sizes_mode == "reduced":
                    n_steps = min(n_steps, len(fc_sizes_reduced))
                if pool_sizes_mode == "frequent":
                    n_steps = min(n_steps, len(pool_sizes_frequent))
                elif pool_sizes_mode == "aggressive":
                    n_steps = min(n_steps, len(pool_sizes_aggressive))
                assert n_steps != 1000

                for i in range(n_steps):
                    if conv_sizes_mode == "reduced":
                        conv_sizes = conv_sizes_reduced[i]
                    else:
                        conv_sizes = conv_sizes_baseline
                    if fc_sizes_mode == "reduced":
                        fc_sizes = fc_sizes_reduced[i]
                    else:
                        fc_sizes = fc_sizes_baseline
                    if pool_sizes_mode == "frequent":
                        pool_sizes = pool_sizes_frequent[i]
                    elif pool_sizes_mode == "aggressive":
                        pool_sizes = pool_sizes_aggressive[i]
                    else:
                        pool_sizes = pool_sizes_baseline
                    configurations_and_counts.append(construct_and_count_one_configuration(
                        conv_sizes=conv_sizes, pool_sizes=pool_sizes, fc_sizes=fc_sizes))

    print(f"[INFO] Generated {len(configurations_and_counts)} model configurations to train.")
    return configurations_and_counts


def main(dir_data, dir_output):

    # load datasets
    print("[INFO] Loading datasets.")
    wave_clus_datasets = load_wave_clus_datasets(os.path.join(dir_data, "wave_clus"))

    # visualize some noise levels
    print("[INFO] Plotting some spike samples.")
    for spike_idx, nl in zip([10, 20, 30, 40], [0.05, 0.1, 0.15, 0.2]):
        fig, axes = plt.subplots(1, 1)
        axes.plot(wave_clus_datasets[("Easy1", nl)].spikes[spike_idx])
        axes.set_title(f"Spike {spike_idx} in diff=Easy1, nl={nl}")
        axes.set_ylim(-1.5, 1.5)
        plt.savefig(os.path.join(dir_output, f"spike_{spike_idx}_in_Easy1_{nl:1.2f}.png"))
        plt.close(fig)

    # run PCA and GMM: identify best number of dimensions to use
    print("[INFO] Finding best n_dim for PCA+GMM clustering using validation datasets.")
    data = []
    n_dims_list = []
    acc_list = []
    best_n_dims = None
    best_total_acc = np.NINF
    for n_dims in range(2, 17):
        total_acc = 0.0
        for (diff, noise) in (("Easy1", 0.15), ("Difficult1", 0.15)):  # same datasets for validation as Li et. al.
            dataset_train, dataset_test = wave_clus_datasets[(diff, noise)].split_train_test(0.5)
            model = PCAGMM(n_dims=n_dims, n_clusters=WaveClusDataset.N_CLUSTERS)
            model.fit(dataset_train.spikes_normalized)
            pred_labels_test = model.predict(dataset_test.spikes_normalized)
            acc = compute_acc(
                pred_labels=pred_labels_test, true_labels=dataset_test.spikes_class,
                n_labels=WaveClusDataset.N_CLUSTERS, test_permutations=True)
            total_acc += acc
            data.append((diff, noise, n_dims, acc))
        if total_acc > best_total_acc:
            best_n_dims = n_dims
            best_total_acc = total_acc
        n_dims_list.append(n_dims)
        acc_list.append(total_acc / 2)
    print(f"[INFO] Best n_dims found to be {best_n_dims} with mean val acc of {(best_total_acc / 2):1.6f}")
    df_results_pcagmm_search = pd.DataFrame(data, columns=["diff", "noise", "n_dims", "acc"])
    df_results_pcagmm_search.to_csv(os.path.join(dir_output, "results_pcagmm_search.csv"), index=False)
    plt.plot(n_dims_list, acc_list)
    plt.xlabel("n_dims")
    plt.ylabel("accuracy")
    plt.show()

    # run PCA and GMM: all datasets
    print("[INFO] Running PCA+GMM clustering using best n_dims for all datasets.")
    data = []
    for (diff, noise), dataset in wave_clus_datasets.items():
        dataset_train, dataset_test = dataset.split_train_test(0.5)
        model = PCAGMM(n_dims=best_n_dims, n_clusters=WaveClusDataset.N_CLUSTERS)
        model.fit(dataset_train.spikes_normalized)
        pred_labels_test = model.predict(dataset_test.spikes_normalized)
        acc = compute_acc(
            pred_labels=pred_labels_test, true_labels=dataset_test.spikes_class,
            n_labels=WaveClusDataset.N_CLUSTERS, test_permutations=True)
        data.append((diff, noise, best_n_dims, acc))
    df_results_pcagmm_all = pd.DataFrame(data, columns=["diff", "noise", "n_dims", "acc"])
    df_results_pcagmm_all.to_csv(os.path.join(dir_output, "results_pcagmm_all.csv"), index=False)

    # train and test CNN architecture from Li et. al.
    # batchnorm and dropout implementation, dropout rate, and learning rate weren't specified in the paper
    # we therefore do a grid search over these parameters to find the best combination
    # using the same two datasets used for hyperparameter optimization in the original paper
    print("[INFO] Searching over hyperparameter combinations for baseline network.")
    data = []
    for batchnorm_mode in ["all", "conv_only", "fc_only", "boundary_only"]:
        for dropout_mode in ["all", "conv_only", "fc_only", "boundary_only"]:
            for dropout_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for learning_rate in [0.0001, 0.001, 0.01]:
                    total_val_acc = 0.0
                    for diff, noise in [("Easy1", 0.15), ("Difficult1", 0.15)]:
                        dataset = wave_clus_datasets[(diff, noise)]
                        # print("[INFO] Constructing and compiling baseline model architecture.")
                        model = construct_cnn1d(
                            n_inputs=64,
                            convolutional_layers=[
                                ("conv", 32),
                                ("conv", 64),
                                ("pool", 2),
                                ("conv", 128),
                                ("pool", 2),
                                ("conv", 128)],
                            fully_connected_layers=[300, 100],
                            n_outputs=WaveClusDataset.N_CLUSTERS,
                            batchnorm_mode=batchnorm_mode,
                            dropout_mode=dropout_mode,
                            dropout_rate=dropout_rate)
                        model.compile(
                            optimizer=k.optimizers.Adam(learning_rate=learning_rate),
                            loss=k.losses.SparseCategoricalCrossentropy(name="loss"),
                            metrics=[k.metrics.SparseCategoricalAccuracy(name="acc")])
                        # print("[INFO] Training baseline model.")
                        dataset_train, dataset_test = dataset.split_train_test(0.5)
                        history = model.fit(
                            x=dataset_train.spikes_normalized,
                            y=dataset_train.spikes_class,
                            batch_size=256,
                            epochs=50,
                            validation_data=(dataset_test.spikes_normalized, dataset_test.spikes_class),
                            callbacks=[
                                k.callbacks.EarlyStopping(monitor="val_acc", patience=4, restore_best_weights=True),
                                k.callbacks.TerminateOnNaN()],
                            verbose=False)
                        total_val_acc += np.max(history.history["val_acc"])
                    data.append((batchnorm_mode, dropout_mode, dropout_rate, learning_rate, total_val_acc))
                    print(f"[INFO] "
                          f"batchnorm_mode={batchnorm_mode}, "
                          f"dropout_mode={dropout_mode}, "
                          f"dropout_rate={dropout_rate}, "
                          f"learning_rate={learning_rate}: "
                          f"total_val_acc={total_val_acc:1.6f}")
    df_results_cnn_baseline = pd.DataFrame(
        data, columns=["batchnorm_mode", "dropout_mode", "dropout_rate", "learning_rate", "total_val_acc"])
    df_results_cnn_baseline.to_csv(os.path.join(dir_output, "results_cnn_baseline.csv"), index=False)

    # perform one final run with the baseline network on all datasets using optimized hyperparameters
    print("[INFO] Training baseline model.")
    batchnorm_mode = "boundary_only"
    dropout_mode = "boundary_only"
    dropout_rate = 0.5
    learning_rate = 0.01
    data = []
    first = True
    for diff, noise in wave_clus_datasets.keys():
        dataset = wave_clus_datasets[(diff, noise)]
        model = construct_cnn1d(
            n_inputs=64,
            convolutional_layers=[
                ("conv", 32),
                ("conv", 64),
                ("pool", 2),
                ("conv", 128),
                ("pool", 2),
                ("conv", 128)],
            fully_connected_layers=[300, 100],
            n_outputs=WaveClusDataset.N_CLUSTERS,
            batchnorm_mode=batchnorm_mode,
            dropout_mode=dropout_mode,
            dropout_rate=dropout_rate)
        model.compile(
            optimizer=k.optimizers.Adam(learning_rate=learning_rate),
            loss=k.losses.SparseCategoricalCrossentropy(name="loss"),
            metrics=[k.metrics.SparseCategoricalAccuracy(name="acc")])
        if first:
            model.summary()
            first = False
        dataset_train, dataset_test = dataset.split_train_test(0.5)
        history = model.fit(
            x=dataset_train.spikes_normalized,
            y=dataset_train.spikes_class,
            batch_size=256,
            epochs=50,
            validation_data=(dataset_test.spikes_normalized, dataset_test.spikes_class),
            callbacks=[
                k.callbacks.EarlyStopping(monitor="val_acc", patience=4, restore_best_weights=True),
                k.callbacks.TerminateOnNaN()],
            verbose=False)
        data.append((diff, noise, np.max(history.history["val_acc"])))
    df_results_cnn_baseline = pd.DataFrame(
        data, columns=["diff", "noise", "val_acc"])
    df_results_cnn_baseline.to_csv(os.path.join(dir_output, "results_cnn_baseline_final.csv"), index=False)

    # try some different configurations for reducing model parameter count
    print("[INFO] Training model variants.")
    batchnorm_mode = "boundary_only"
    dropout_mode = "boundary_only"
    dropout_rate = 0.5
    learning_rate = 0.01
    configurations_and_counts = construct_and_count_all_configurations()
    data_outer = []
    for i, (configuration, n_mult) in enumerate(configurations_and_counts):
        data = []
        total_val_acc = 0.0
        n_params = None
        for diff, noise in wave_clus_datasets.keys():
            dataset = wave_clus_datasets[(diff, noise)]
            model = construct_cnn1d(
                n_inputs=64,
                **configuration,
                n_outputs=WaveClusDataset.N_CLUSTERS,
                batchnorm_mode=batchnorm_mode,
                dropout_mode=dropout_mode,
                dropout_rate=dropout_rate)
            model.compile(
                optimizer=k.optimizers.Adam(learning_rate=learning_rate),
                loss=k.losses.SparseCategoricalCrossentropy(name="loss"),
                metrics=[k.metrics.SparseCategoricalAccuracy(name="acc")])
            n_params = model.count_params()
            dataset_train, dataset_test = dataset.split_train_test(0.5)
            history = model.fit(
                x=dataset_train.spikes_normalized,
                y=dataset_train.spikes_class,
                batch_size=256,
                epochs=50,
                validation_data=(dataset_test.spikes_normalized, dataset_test.spikes_class),
                callbacks=[
                    k.callbacks.EarlyStopping(monitor="val_acc", patience=4, restore_best_weights=True),
                    k.callbacks.TerminateOnNaN()],
                verbose=False)
            data.append((str(configuration), n_params, n_mult, diff, noise, np.max(history.history["val_acc"])))
            total_val_acc += np.max(history.history["val_acc"])
        df_results_cnn_configuration = pd.DataFrame(
            data, columns=["configuration", "n_params", "n_mult", "diff", "noise", "val_acc"])
        df_results_cnn_configuration.to_csv(
            os.path.join(dir_output, f"results_cnn_configuration_{i}.csv"), index=False)
        data_outer.append((str(configuration), n_params, n_mult, total_val_acc))
        print(f"[INFO] config #{i}: n_params={n_params}, n_mult={n_mult}, total_val_acc={total_val_acc:1.6f} ")
    df_results_cnn_configurations_all = pd.DataFrame(
        data_outer, columns=["configuration", "n_params", "n_mult", "total_val_acc"])
    df_results_cnn_configurations_all.to_csv(
        os.path.join(dir_output, f"results_cnn_configurations_all.csv"), index=False)


if __name__ == "__main__":
    _dir_data = r"D:\Documents\Academics\BME517\final_project\data"
    _dir_output = r"D:\Documents\Academics\BME517\final_project\output"
    main(_dir_data, _dir_output)
