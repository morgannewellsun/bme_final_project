from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class PCAGMM:

    def __init__(self, n_dims: int, n_clusters: int):
        self.n_dims: int = n_dims
        self.n_clusters: int = n_clusters
        self.pca: PCA = PCA(n_components=n_dims)
        self.gmm: GaussianMixture = GaussianMixture(n_components=n_clusters)

    def fit(self, spikes_normalized):
        spikes_reduced = self.pca.fit_transform(spikes_normalized)
        self.gmm.fit(spikes_reduced)

    def predict(self, spikes_normalized):
        spikes_reduced = self.pca.transform(spikes_normalized)
        return self.gmm.predict(spikes_reduced)
