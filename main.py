import dataCleaner
import correlation
import modeling
import clustering

if __name__ == "__main__":

    # Data Preparation & Quality Assurance
    dataCleaner.cleanData()

    # Feature Engineering & Selection
    correlation.indexCorrelation()

    # Modeling
    modeling.modeling()

    # Clustering
    clustering.clustering()
