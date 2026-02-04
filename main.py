import dataCleaner
import correlation

if __name__ == "__main__":

    # Data Preparation & Quality Assurance
    dataCleaner.cleanData()

    # Feature Engineering & Selection
    correlation.indexCorrelation()
