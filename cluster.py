from dataprep import data_r
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    silhouette_score
)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False
})

# data prep
temp = data_r.drop(columns=["Mg", "log_10(Mdyn)", "log_10(Mbary)","log_10(Mgas)","log_10(M*)"])
X = temp.iloc[:, 1:-1].values
y = temp.iloc[:, -1].values

X = StandardScaler().fit_transform(X)


# accuracy evaluation
def clustering_accuracy(true_labels, cluster_labels):
    true_labels = np.asarray(true_labels)
    cluster_labels = np.asarray(cluster_labels)

    labels_true = np.unique(true_labels)
    labels_pred = np.unique(cluster_labels)

    cost = np.zeros((len(labels_pred), len(labels_true)), dtype=int)

    for i, p in enumerate(labels_pred):
        for j, t in enumerate(labels_true):
            cost[i, j] = np.sum((cluster_labels == p) & (true_labels != t))

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {labels_pred[r]: labels_true[c] for r, c in zip(row_ind, col_ind)}

    aligned_pred = np.array([mapping[c] for c in cluster_labels])

    accuracy = np.mean(aligned_pred == true_labels)

    return accuracy, aligned_pred


# tuning
linkages = ["average"]#, "average", "complete"]
metrics  = ["cosine"] #", "manhattan", "cosine"]
cluster_counts = [2, 3, 4, 5]

results = []

for lk in linkages:
    for nc in cluster_counts:
        for dm in metrics:

            # Ward only supports Euclidean
            if lk == "ward" and dm != "euclidean":
                continue

            try:
                model = AgglomerativeClustering(
                    n_clusters=nc,
                    linkage=lk,
                    metric=dm
                )

                labels = model.fit_predict(X)

                acc, _ = clustering_accuracy(y, labels)
                ari = adjusted_rand_score(y, labels)
                nmi = normalized_mutual_info_score(y, labels)
                hom = homogeneity_score(y, labels)
                com = completeness_score(y, labels)
                sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else None

                results.append({
                    "linkage": lk,
                    "metric": dm,
                    "clusters": nc,
                    "accuracy": acc,
                    "ARI": ari,
                    "NMI": nmi,
                    "homogeneity": hom,
                    "completeness": com,
                    "silhouette": sil
                })

            except:
                pass


# sort results
results_sorted = sorted(results, key=lambda x: x["ARI"], reverse=True)


print("\nTop 10 configurations (by ARI):")
for r in results_sorted[:10]:
    print(r)


# Dendogram of best model
best = results_sorted[0]
print("\nBest configuration:", best)

Z = linkage(
    X,
    method=best["linkage"],
    metric=best["metric"]
)

from scipy.cluster.hierarchy import fcluster

clusters_4 = fcluster(Z, t=4, criterion='maxclust')

# clusters_4 now contains the cluster labels determined by the dendrogram cut
print("Unique clusters at this cut:", np.unique(clusters_4))


plt.figure(figsize=(14, 6))
dendrogram(
    Z,
    truncate_mode="level",
    p=12,
    leaf_rotation=90
)
plt.title(
    f"Dendrogram (Linkage: {best['linkage']}, Metric: {best['metric']}, Best clusters={best['clusters']})"
)

plt.xticks([])
plt.tick_params(axis='x', which='both', bottom=False)

# Add a simple label instead
plt.xlabel("Clusters")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(f"dendogram.png", dpi=400, bbox_inches="tight")
plt.close()

acc_4d, aligned_4d = clustering_accuracy(y, clusters_4)

ARI_4d = adjusted_rand_score(y, clusters_4)
NMI_4d = normalized_mutual_info_score(y, clusters_4)
HOM_4d = homogeneity_score(y, clusters_4)
COMP_4d = completeness_score(y, clusters_4)
SIL_4d = silhouette_score(X, clusters_4) 

print("Dendrogram-cut 4-cluster metrics:")
print("Accuracy:", acc_4d)
print("ARI:", ARI_4d)
print("NMI:", NMI_4d)
print("Homogeneity:", HOM_4d)
print("Completeness:", COMP_4d)
print("Silhouette:", SIL_4d)
