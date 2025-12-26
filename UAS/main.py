import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set visualization style
sns.set(style="whitegrid")

# 1. LOAD & CLEAN DATA
df = pd.read_csv('llm_comparison_dataset.csv')
df.columns = df.columns.str.strip()  # Clean whitespace from column names
df = df.drop_duplicates()           # Remove potential duplicate rows

# 2. DESCRIPTIVE STATISTICS
desc_stats = df.describe()
desc_stats.to_csv('full_descriptive_statistics.csv')

# 3. RELATIONS (CORRELATION HEATMAP)
plt.figure(figsize=(14, 10))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of LLM Metrics', fontsize=16)
plt.tight_layout()
plt.savefig('full_correlation_heatmap.png')

# 4. OPTIMIZED CLUSTERING (K-MEANS)
# Selecting key features for segmentation
cluster_features = ['Benchmark (MMLU)', 'Benchmark (Chatbot Arena)', 'Price / Million Tokens', 'Speed (tokens/sec)']
X_cluster = df[cluster_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Optimized k selection via Elbow Method
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method for Optimal Clusters', fontsize=14)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.savefig('full_elbow_plot.png')

# Final Clustering with optimal k=4
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

plt.figure(figsize=(12, 7))
sns.scatterplot(data=df, x='Benchmark (MMLU)', y='Price Rating',
                hue='Cluster', style='Open-Source', palette='viridis', s=100, alpha=0.8)
plt.title('LLM Segments: MMLU vs Chatbot Arena', fontsize=16)
plt.legend(title='Cluster & Source', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('full_clustering_chart.png')

# 5. OPTIMIZED REGRESSION (RANDOM FOREST)
# Predicting human-rated Chatbot Arena score using technical metrics
features_reg = ['Benchmark (MMLU)', 'Speed (tokens/sec)', 'Latency (sec)',
                'Price / Million Tokens', 'Open-Source', 'Context Window',
                'Training Dataset Size', 'Compute Power']
target_reg = 'Price Rating'

X = df[features_reg]
y = df[target_reg]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Feature Importance Visualization
feat_importance = pd.DataFrame({'Feature': features_reg, 'Importance': rf_model.feature_importances_})
feat_importance = feat_importance.sort_values(by='Importance', ascending=False)

print(f"Regresian model feature importance: \n{feat_importance.to_string(index=False)}")

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importance, palette='magma')
plt.title('Feature Importance for Predicting Chatbot Arena Score', fontsize=14)
plt.tight_layout()
plt.savefig('full_feature_importance.png')

# 6. EXPORT FINAL RESULTS
df.to_csv('fully_analyzed_llm_data.csv', index=False)
print("Analysis complete. Charts and CSV files generated.")