from sklearn.metrics import silhouette_samples, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import utils
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
        ): {"name": "KMeans + 5D PCA"},
    }

    bestScore = float("inf")
    bestLabels = None

    for model, modelInfo in models.items():

        # Fit the model and get the labels
        model.fit(data)
        labels = model.named_steps["cluster"].labels_

        assert len(np.unique(labels)) == 3, "The model did not create 3 clusters"

        score = davies_bouldin_score(data, labels)

        print(f"{modelInfo['name']} score: {score:.4f}")

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

    return clusters


if __name__ == "__main__":
    clustering()
