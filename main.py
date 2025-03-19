import pandas as pd

from data_processing.data_loader import DataLoader
from data_processing.data_preprocessor import DataPreprocessor
from data_processing.cluster_analysis import ClusterAnalysis

# Загрузка данных
loader = DataLoader('data/BGG_Data_Set.csv')
data = loader.load_data()
print(data.info())
print(data)
print(data.iloc[0])

# Предварительная обработка данных
preprocessor = DataPreprocessor(data)
preprocessor.clean_data()
preprocessor.transform_data()
data = preprocessor.data

# Нормализация данных
features = ['Rating Average', 'Owned Users']
X_scaled = preprocessor.normalize_data(features)

# Кластерный анализ
cluster_analysis = ClusterAnalysis(data)
cluster_analysis.elbow_method(X_scaled)

optimal_ks = [3, 4, 5]
for k in optimal_ks:
    result = cluster_analysis.perform_clustering(X_scaled, k)
    full_result = pd.concat([
        preprocessor.removed_data.reset_index(drop=True),
        result.reset_index(drop=True)],
        axis=1)
    cluster_analysis.visualize_clusters()
    export_result = full_result.sort_values(by=['Cluster'], ascending=False)
    export_result.to_excel(f'output_{k}.xlsx', index=False)
