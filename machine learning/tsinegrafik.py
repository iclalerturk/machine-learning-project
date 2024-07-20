import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def read_weights(filename):
    data = pd.read_csv(filename)
    weights = data.iloc[:, 1:].values
    return weights

# Read the weight values
wGD_weights = read_weights('wGD_values.csv')
wSGD_weights = read_weights('wSGD_values.csv')
wADAM_weights = read_weights('wADAM_values.csv')

num_epochs = wGD_weights.shape[0]
num_weights = wGD_weights.shape[1]

# Concatenate the weights for t-SNE
all_weights = np.concatenate((wGD_weights, wSGD_weights, wADAM_weights), axis=0)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
weights_2d = tsne.fit_transform(all_weights)

# Split the transformed weights
wGD_2d = weights_2d[:num_epochs]
wSGD_2d = weights_2d[num_epochs:2*num_epochs]
wADAM_2d = weights_2d[2*num_epochs:]

# Plot the trajectories
plt.figure(figsize=(12, 8))
plt.plot(wGD_2d[:, 0], wGD_2d[:, 1], label='GD Trajectory', marker='o', linestyle='-')
plt.plot(wSGD_2d[:, 0], wSGD_2d[:, 1], label='SGD Trajectory', marker='x', linestyle='--')
plt.plot(wADAM_2d[:, 0], wADAM_2d[:, 1], label='ADAM Trajectory', marker='^', linestyle=':')

plt.scatter(wGD_2d[:, 0], wGD_2d[:, 1], c='blue', s=50, alpha=0.5, edgecolors='w')
plt.scatter(wSGD_2d[:, 0], wSGD_2d[:, 1], c='green', s=50, alpha=0.5, edgecolors='w')
plt.scatter(wADAM_2d[:, 0], wADAM_2d[:, 1], c='red', s=50, alpha=0.5, edgecolors='w')

plt.legend()
plt.title('Weight Trajectories Using t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
