from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import utils
import os


def clustering() -> None:
    """

    Args:
        - None

    Returns:
        - None
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

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(cleanData)

    centersDF = pd.DataFrame(kmeans.cluster_centers_, columns=cleanData.columns)

    # Group the centers
    for col in centersDF.columns.to_list():
        # Group the centers by the original column name
        originalCol = col.split("_")[0]

        if originalCol not in centersDF.columns:
            gcols = [c for c in centersDF.columns if c.startswith(originalCol + "_")]

            centersDF[originalCol] = centersDF[gcols].apply(
                lambda x: np.array(x), axis=1
            )
            centersDF.drop(columns=gcols, inplace=True)

    # 5 most important features for each cluster
    for i in range(3):
        print(f"Cluster {i}:")
        clusterCenter = centersDF.iloc[i]

        # Do the magnitude of each feature and get the top 3
        topFeatures = (
            clusterCenter.apply(np.linalg.norm).sort_values(ascending=False)[:3].index
        )

        for t in topFeatures.to_list():
            print(f"\t{t}")

    # Save the centers to a CSV file
    centersDF.to_csv(os.path.join(dataFolder, "centers.csv"), index=False)


if __name__ == "__main__":
    clustering()
