import numpy as np
from scipy.io import loadmat


class WaveClusDataset:
    N_CLUSTERS = 3

    def __init__(self):
        self.samples_per_spike: int = 0
        self.spikes_class: np.ndarray = np.array([])
        self.spikes: np.ndarray = np.array([])
        self.spikes_normalized: np.ndarray = np.array([])
        self.normalization_mean: np.ndarray = np.array([])
        self.normalization_std: np.ndarray = np.array([])

    def load(self, path, samples_per_spike: int = 64, verbose: bool = False):

        self.samples_per_spike = samples_per_spike
        dataset = loadmat(path)

        if "data" not in dataset.keys():
            raise RuntimeError("[ERROR] data not in dataset keys.")
        if np.product(dataset["data"].shape) != np.max(dataset["data"].shape):
            raise RuntimeError(f"[ERROR] data has shape {dataset['data'].shape}.")
        np_raw_data = dataset["data"].flatten()
        print(f"[INFO] Dataset contains {np_raw_data.shape[0]} samples.") if verbose else None

        if "spike_times" not in dataset.keys():
            raise RuntimeError("[ERROR] spike_times not in dataset keys.")
        if len(dataset["spike_times"]) != 1:
            raise RuntimeError("[ERROR] spike_times doesn't have a length of 1.")
        if len(dataset["spike_times"][0]) != 1:
            raise RuntimeError("[ERROR] spike_times[0] doesn't have a length of 1.")
        if np.product(dataset["spike_times"].shape) != np.max(dataset["spike_times"].shape):
            raise RuntimeError(f"[ERROR] spike_times has shape {dataset['spike_times'].shape}.")
        spikes_time = dataset["spike_times"][0][0].flatten()
        if _max_spike_time := np.max(spikes_time) + self.samples_per_spike > np_raw_data.shape[0]:
            raise RuntimeError(f"[ERROR] spike_times contains a spike starting at time {_max_spike_time}.")

        if "spike_class" not in dataset.keys():
            raise RuntimeError("[ERROR] spike_class not in dataset keys.")
        if len(dataset["spike_class"]) != 1:
            raise RuntimeError("[ERROR] spike_class doesn't have a length of 1.")
        if len(dataset["spike_class"][0]) != 3:
            raise RuntimeError("[ERROR] spike_class[0] doesn't have a length of 3.")
        if np.product(dataset["spike_class"].shape) != np.max(dataset["spike_class"].shape):
            raise RuntimeError(f"[ERROR] spike_class has shape {dataset['spike_class'].shape}.")
        self.spikes_class = dataset["spike_class"][0][0].flatten() - 1
        if _unique_classes := set(np.unique(self.spikes_class)) != set(range(WaveClusDataset.N_CLUSTERS)):
            raise RuntimeError(f"[ERROR] spike_class contains values {_unique_classes}.")

        if spikes_time.shape[0] != self.spikes_class.shape[0]:
            raise ValueError(
                f"[ERROR] {spikes_time.shape[0]} spike times but {self.spikes_class.shape[0]} spike classes.")

        self.spikes = np.array(
            [np_raw_data[time:time + self.samples_per_spike] for time in spikes_time])
        print(f"[INFO] Dataset contains {spikes_time.shape[0]} spikes.") if verbose else None

        self.normalization_mean = np.mean(self.spikes, axis=0, keepdims=True)
        self.spikes_normalized = self.spikes - self.normalization_mean
        self.normalization_std = np.std(self.spikes_normalized, axis=0, keepdims=True)
        self.spikes_normalized = self.spikes_normalized / self.normalization_std

    def __len__(self):
        return self.spikes.shape[0]

    def split_train_test(self, train_ratio: float):

        if not (0 <= train_ratio <= 1):
            raise ValueError(f"[ERROR] train_ratio must be between 0 and 1, got value of {train_ratio}.")

        split_idx = int(len(self) * train_ratio)
        train_dataset = WaveClusDataset()
        train_dataset.samples_per_spike = self.samples_per_spike
        train_dataset.spikes_class = self.spikes_class[:split_idx]
        train_dataset.spikes = self.spikes[:split_idx]
        test_dataset = WaveClusDataset()
        test_dataset.samples_per_spike = self.samples_per_spike
        test_dataset.spikes_class = self.spikes_class[split_idx:]
        test_dataset.spikes = self.spikes[split_idx:]

        normalization_mean = np.mean(train_dataset.spikes, axis=0, keepdims=True)
        train_dataset.normalization_mean = normalization_mean
        train_dataset.spikes_normalized = train_dataset.spikes - normalization_mean
        test_dataset.normalization_mean = normalization_mean
        test_dataset.spikes_normalized = test_dataset.spikes - normalization_mean

        normalization_std = np.std(train_dataset.spikes, axis=0, keepdims=True)
        train_dataset.normalization_std = normalization_std
        train_dataset.spikes_normalized = train_dataset.spikes_normalized / normalization_std
        test_dataset.normalization_std = normalization_std
        test_dataset.spikes_normalized = test_dataset.spikes_normalized / normalization_std

        return train_dataset, test_dataset
