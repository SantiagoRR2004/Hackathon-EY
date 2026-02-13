from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_samples, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils
import json
import os


def bestClusteringModel(data: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the best clustering model. It returns the same
    data with a new column "Cluster".

    Args:
        - data (pd.DataFrame): The data to cluster

    Returns:
        - pd.DataFrame: The data with a new column "Cluster"
    """
    models = {
        Pipeline(steps=[("cluster", KMeans(n_clusters=3, random_state=42))]): {
            "name": "KMeans"
        },
        Pipeline(
            steps=[
                ("pca", PCA(n_components=5, random_state=42)),
                ("cluster", KMeans(n_clusters=3, random_state=42)),
            ]
        ): {"name": "5D PCA + KMeans"},
        Pipeline(steps=[("cluster", AgglomerativeClustering(n_clusters=3))]): {
            "name": "Agglomerative Clustering"
        },
        Pipeline(
            steps=[
                ("pca", PCA(n_components=0.9, random_state=42)),
                ("cluster", AgglomerativeClustering(n_clusters=3)),
            ]
        ): {"name": "90% PCA + Agglomerative Clustering"},
        Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                ("pca", PCA(n_components=0.95, random_state=42)),
                (
                    "cluster",
                    KMeans(n_clusters=3, n_init=20, max_iter=500, random_state=42),
                ),
            ]
        ): {"name": "RobustScaler + 95% PCA + KMeans"},
        Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=10, random_state=42)),
                (
                    "cluster",
                    KMeans(n_clusters=3, n_init=20, max_iter=500, random_state=42),
                ),
            ]
        ): {"name": "StandardScaler + 10D PCA + KMeans"},
        Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                ("pca", PCA(n_components=0.9, random_state=42)),
                ("cluster", AgglomerativeClustering(n_clusters=3, linkage="ward")),
            ]
        ): {"name": "RobustScaler + 90% PCA + Agglomerative (Ward)"},
        Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=8, random_state=42)),
                ("cluster", AgglomerativeClustering(n_clusters=3, linkage="average")),
            ]
        ): {"name": "StandardScaler + 8D PCA + Agglomerative (Average)"},
        Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                ("svd", TruncatedSVD(n_components=10, random_state=42)),
                ("cluster", MiniBatchKMeans(n_clusters=3, n_init=20, random_state=42)),
            ]
        ): {"name": "RobustScaler + SVD + MiniBatch KMeans"},
        Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=42)),
                (
                    "cluster",
                    GaussianMixture(
                        n_components=3, covariance_type="full", random_state=42
                    ),
                ),
            ]
        ): {"name": "StandardScaler + 95% PCA + Gaussian Mixture Model"},
    }

    bestScore = float("inf")
    bestLabels = None

    pcaData = PCA(n_components=2, random_state=42).fit_transform(data)

    for model, modelInfo in models.items():

        # Fit the model and get the labels
        model.fit(data)

        try:
            labels = model.named_steps["cluster"].labels_
        except Exception as e:
            labels = model.predict(data)

        if len(np.unique(labels)) != 3:
            print(f"{modelInfo['name']} did not create 3 clusters.")

        else:
            score = davies_bouldin_score(data, labels)

            print(f"{modelInfo['name']} score: {score:.4f}")

            # Plot the clusters
            utils.graphClusters(pcaData, labels, title=modelInfo["name"])

            # Update the best
            if score < bestScore:
                bestScore = score
                bestLabels = labels

    # Add the best labels to the data
    data["Cluster"] = bestLabels

    return data


def clustering() -> list:
    """
    Create 3 clusters of the data and return the top 3 features for each cluster.

    These features are found by calculating the silhouette score
    for each of them and taking the top 3 for each cluster.

    Args:
        - None

    Returns:
        - list: A list of lists of the top 3 features for each cluster
    """
    # Get the paths
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataFolder = os.path.join(currentDirectory, "data")

    # Load the data
    cleanData = utils.loadCSV(os.path.join(dataFolder, "mental_healthCleaned.csv"))

    # Load the indexes
    with open(os.path.join(dataFolder, "indexes.json"), "r") as f:
        indexes = json.load(f)

    validColumns = [v for k, v in indexes.items() if k != "Other"]
    validColumns = set([col for cols in validColumns for col in cols])

    cleanData = cleanData[[col for col in cleanData.columns if col in validColumns]]

    # Divide columns that are vectors into their components
    for col in cleanData.columns:
        cleanData = cleanData.join(
            pd.DataFrame(cleanData.pop(col).tolist()).add_prefix(f"{col}_")
        )

    utils.printSeparator("Clustering")

    cleanData = bestClusteringModel(cleanData)

    # Group the rows
    for col in cleanData.columns.to_list():
        # Group the centers by the original column name
        originalCol = col.split("_")[0]

        if originalCol not in cleanData.columns:
            gcols = [c for c in cleanData.columns if c.startswith(originalCol + "_")]

            cleanData[originalCol] = cleanData[gcols].apply(
                lambda x: np.array(x), axis=1
            )
            cleanData.drop(columns=gcols, inplace=True)

    utils.printSeparator("Clusters")

    silhouette = pd.DataFrame()

    for col in cleanData.columns:
        if col != "Cluster":
            silhouette[col] = silhouette_samples(
                np.vstack(cleanData[col].values), cleanData["Cluster"]
            )
        else:
            silhouette[col] = cleanData[col]

    clusters = []

    # 5 most important features for each cluster
    for i in range(3):
        print(f"Cluster {i}:")
        topFeatures = {}

        for col in silhouette.columns:
            if col not in ["Cluster"]:
                topFeatures[col] = silhouette[col][silhouette["Cluster"] == i].mean()

        # Sort the features by silhouette score
        topFeatures = sorted(topFeatures.items(), key=lambda x: x[1], reverse=True)

        # Graph the top 5 features for this cluster
        utils.plotTopCorrelatedQuestions(
            topFeatures[:5],
            dataFolder,
            title=f"Cluster {i} Silhouette Scores",
            versus=False,
        )

        for feature, score in topFeatures[:3]:
            print(f"\t{feature}")

        clusters.append([feature for feature, score in topFeatures[:3]])

    # Load the original data
    originalData = pd.read_csv(os.path.join(dataFolder, "mental_health.csv"))
    originalData = originalData[
        [col for col in originalData.columns if col in validColumns]
    ]

    # For each column graph the distribution of the values for each cluster
    for col in originalData.columns:

        colData = originalData[col].fillna("NaN")

        # Unique values lower than 10
        if colData.nunique() < 10:
            freq = pd.crosstab(colData, cleanData["Cluster"], dropna=False)
            matrix = freq.values

            fig, ax = plt.subplots(figsize=(len(freq.columns), len(freq.index)))
            im = ax.imshow(matrix, aspect="equal", cmap="coolwarm")
            plt.title(col)

            # Add numbers inside the cells
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]}",
                        ha="center",
                        va="center",
                        color="black",
                    )

            # Set y-axis labels
            ax.set_yticks(range(len(freq.index)))
            ax.set_yticklabels(freq.index)

            # Remove x-axis labels
            ax.set_xticks([])
            plt.xlabel("Cluster")
            ax.grid(False)

            plt.savefig(
                os.path.join(
                    dataFolder, f"Clustering{col.replace(' ', '').replace('/', '')}.png"
                ),
                bbox_inches="tight",
            )
            plt.close()

    return clusters


if __name__ == "__main__":
    clustering()
