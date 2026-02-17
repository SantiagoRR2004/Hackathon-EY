# Predictive Model for Mental Health Risk

This repository was created for a [Hackathon organized by EY](https://challenge.ey.com/challenges/innovating-mental-health-risk-assessment). The explanation for what we had to do is [this pdf](./Use%20Case%20challenge.pdf). This is the documentation we submitted:

## Introduction

Mental health represents one of the most significant public health and social challenges worldwide. Early identification of potential mental health risks can enable preventive interventions and contribute to improved well-being and quality of life.

The objective of this project is to develop and evaluate a Machine Learning–based system capable of identifying patterns associated with mental health risk using self-reported data. The approach combines data preprocessing, feature engineering and selection, supervised learning models, and unsupervised clustering techniques.

The file that runs the whole program is [main.py](../main.py).

## Data Preparation & Quality Assurance

The first stage of the project focuses on ensuring the quality, consistency, and reliability of the dataset. The most important factor was to keep the same features because the csv submission requires that the columns be named the same. The solution we chose for this was to keep all the data from a single column inside a vector. This way we wouldn’t have to compress all information to a single value or use complicated maths to calculate the effect of multiple new columns to the old one.

The file [dataCleaner.py](../dataCleaner.py) is the one that has all the functions that clean columns. There are individual functions that clean different types of columns.

For most of the categorical questions we just use one-hot encoding, but some have some ordinal categories. For those we manually map those categories to the range 0 to 1 to keep the order and apply one-hot to the other options. The most important fact is that we consider NaN as another category because it is important to know who hasn't answered and there are too many NaNs to remove or calculate replacements.

The other important columns were the free form text ones of `Why or why not?` and `Why or why not?.1`. We used a SentenceTransformer to obtain the embeddings and then applied PCA to get 90% of the variance. However, initially we used the length of the response and there was still a high correlation.

For purely numerical columns we used _MinMaxScaler_ if they had no outliers and _RobustScaler_ if they had.

The final result is stored in [mental_healthCleaned.csv](../data/mental_healthCleaned.csv).

## Feature Engineering & Selection

For this section we had to choose which questions corresponded to each index. This is important because the next sections require the use of these indexes. We stored the indexes in the file [indexes.json](../data/indexes.json).

### Correlation Analysis

The function to calculate the correlation of two columns is in [correlation.py](../correlation.py). We use something that could be called a Procrustes Similarity.

This is calculated by using the singular values of a SVD to the cross-covariance matrix of both columns. We have tested that this similarity is always between 0 and 1 and the column order inside each feature doesn’t affect the result. We tested this experimentally in the file [testCorrelation.py](../testCorrelation.py).

All the correlations are in the image [CorrelationAll](../data/CorrelationAll.png) but here are the correlations for each of the indexes:

<img src="../data/CorrelationMentalHealthSupportIndex.png"/>

<img src="../data/CorrelationOrganizationalOpennessScore.png"/>

<img src="../data/CorrelationWorkplaceStigmaIndex.png"/>

## Modeling

The code is in the file [modeling.py](../modeling.py). Supervised learning techniques were applied to predict variables related to mental health outcomes. An initial 80/20 train–test split was employed, but then a 5-fold cross-validation strategy (K-Fold with shuffling and a fixed random seed) was adopted to obtain more robust and generalizable performance estimates.

F1-score was chosen as the main evaluation metric. We used multiple models and chose the model that gave the highest F1-score from the cross validation.

The best model for `Do you currently have a mental health disorder?` was Logistic Regression with an F1-score of 0.4683.

<img src="../data/Top10Doyoucurrentlyhaveamentalhealthdisorder%3F.png"/>

The best model for `Have you ever sought treatment for a mental health issue from a mental health professional?` was Logistic Regression with an F1-score of 0.7353.

<img src="../data/Top10Haveyoueversoughttreatmentforamentalhealthissuefromamentalhealthprofessional%3F.png"/>

## Clustering

The code is in the file [clustering.py](../clustering.py). To obtain the best clustering model we tried as many clustering algorithms as we could and chose the one that had the best Davies-Bouldin score. The problem is that it seems that there are 4 distinct clusters when we apply PCA, so most models have the same score.

<img src="../data/ClustersKMeans.png"/>

Once we had the clusters we had to choose how to calculate the most important features for each cluster. Because in our case each feature is a column, we can simply apply Silhouette to each feature in each cluster. In the data folder we show the top 5 features for each cluster. These are for cluster 2:

<img src="../data/Top5Cluster2SilhouetteScores.png"/>

## Results and Discussion

We ran out of time to really process the results we obtained. Our main focus was on submitting good csv that granted us better scores, but we will try to explain the results anyway. All the graphs are in the [data folder](../data/).

For the [Mental Health Support Index](../data/CorrelationMentalHealthSupportIndex.png) we can observe that there are clearly two blocks of questions. Current employer versus previous employer. The way an employer handles mental health is related to all other ways an employer handles mental health.

For the [Organizational Openness Score](../data/CorrelationOrganizationalOpennessScore.png) we can see pairs of related questions more clearly.

For the [Workplace Stigma Index](../data/CorrelationWorkplaceStigmaIndex.png) we can also observe the division between current and previous employers.

For the clustering we created graphs that show how the different answers to each question are divided among clusters. We didn’t have time to analyze each question, but here is an example:

<img src="../data/ClusteringDidyouhearoforobservenegativeconsequencesforco-workerswithmentalhealthissuesinyourpreviousworkplaces%3F.png"/>

We can see that the second cluster 2 has most of the non-empty answers.

A very important conclusion to take from this dataset is that AI specialists should be part of all stages in the data pipeline. They would recommend making choosing an option mandatory in all categorical questions. This would provide better results and is clearly possible because some questions have no NaN answers.

Some forms like Google forms allow limits on numerical questions. This should have been applied to the age question, but we solved it with the _Robust Scaler_.

A different correlation we could have applied was Normalized Mutual Information. It changed most of the submission answers but we had a submission limit.
