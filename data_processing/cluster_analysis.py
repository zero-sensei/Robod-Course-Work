import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class ClusterAnalysis:
    def __init__(self, data):
        self.data = data

    def elbow_method(self, X_scaled, max_k=10):
        inertia = []
        K = range(1, max_k + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        plt.plot(K, inertia, 'bx-')
        plt.xlabel('Количество кластеров')
        plt.title('Метод локтя для выбора количества кластеров')
        plt.show()

    def perform_clustering(self, X_scaled, optimal_k):
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        self.data['Cluster'] = kmeans.fit_predict(X_scaled)
        return self.data

    def visualize_clusters(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data['Owned Users'], self.data['Rating Average'], c=self.data['Cluster'], cmap='viridis', marker='o', alpha=0.7)
        plt.title('')
        plt.xlabel('Owned Users')
        plt.ylabel('Rating Average')
        plt.colorbar(label='Кластер')
        plt.grid(True)
        plt.show()
