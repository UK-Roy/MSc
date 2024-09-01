import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("C:/Users/tanveer/thesis/safety-gymnasium-main/MSc/code/env.csv")
scaler = StandardScaler()

states = df.drop(['action 0', 'action 1', 'label'], axis='columns', inplace=False).to_numpy()
actions = df[['action 0', 'action 1']].to_numpy()
label = df['label'].to_numpy().astype(int)

state_action = df.drop('label', axis='columns', inplace=False)
lidar_action = state_action.drop([f"state {i}" for i in range(0, 40)], axis='columns', inplace=False)
lidar = lidar_action.drop([f"action {i}" for i in range(0, 2)], axis='columns', inplace=False).to_numpy()
scaler.fit(lidar)
feature = scaler.transform(lidar)
# Combine states and actions into a single feature array
# X = np.hstack((states, actions))

print("Shape of feature vector X:", state_action.shape)
print(f"label: {label[100:120]}")
# Define the number of clusters
n_clusters = 2

# Initialize and fit the KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=3)
kmeans.fit(feature)

# Print out the cluster centers
# print("Cluster Centers:")
# print(kmeans.cluster_centers_)

# Assign each state-action pair to a cluster
clusters = kmeans.predict(feature)

# Print the first few clusters to see how the data is grouped
print("First 10 data points and their clusters:")
for i in range(100, 120):
    print(f"{i}. Cluster: {clusters[i]}")


# Plot the rewards for each cluster
# cluster_rewards = []
# for i in range(n_clusters):
#     cluster_rewards.append(np.mean(rewards[clusters == i]))

# plt.bar(range(n_clusters), cluster_rewards)
# plt.xlabel('Cluster')
# plt.ylabel('Average Reward')
# plt.title('Average Reward by Cluster')
# plt.show()

accuracy = (clusters == label).mean()

print(f"Accuracy: {accuracy}")
# Save the trained KMeans model
joblib.dump(kmeans, 'kmeans_model.pkl')
print("Model is saved")