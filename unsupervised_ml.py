import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Preparing the customer data
# Monthly Spending (KES) and Monthly Visits
customers_data = [
    [1000, 2],
    [1500, 3],
    [2000, 2],
    [5000, 6],
    [6000, 7],
    [5500, 5],
    [12000, 12],
    [15000, 14],
    [13000, 11],
    [3000, 4]
]

# Using StandardScaler to scale the spending and visits features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customers_data)

# Create the training model for KMeans
model = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train the model
model.fit(X_scaled)

# Displaying the cluster assigned to each customer
labels = model.labels_
print(" Customer clustered group Assigned: ")
for i in range(len(customers_data)):
    print(f"Customer {i + 1} belongs to Cluster {labels[i]}")

# Display the total number of customers in each cluster
print(" Total customers for each clustered Group: ")
cluster_counts = {0: 0, 1: 0, 2: 0}
for Group_id in labels:
    cluster_counts[Group_id] += 1

for Group_id, count in cluster_counts.items():
    print(f"Cluster group {Group_id}: {count} customer(s)")

spending = [row[0] for row in customers_data]
visits = [row[1] for row in customers_data]

plt.figure(figsize=(8, 6))
# Create the scatter plot using labels to control the point colors
scatter = plt.scatter(spending, visits, c=labels, cmap='viridis', s=100, edgecolors='black')

# Labeling and styling the plot
plt.title('Supermarket Customer Segmentation (K-Means)')
plt.xlabel('Monthly Spending (KES)')
plt.ylabel('Monthly Visits')
plt.grid(True, linestyle='--', alpha=0.5)

# Add a color legend identifying the clusters
plt.colorbar(scatter, label='Cluster ID', ticks=[0, 1, 2])

# Render the graph window
plt.show()
