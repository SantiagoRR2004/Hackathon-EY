import dataCleaner
import correlation
import modeling
import clustering
import json

if __name__ == "__main__":

    # Data Preparation & Quality Assurance
    dataCleaner.cleanData()

    # Feature Engineering & Selection
    groups = correlation.indexCorrelation()

    # Modeling
    modelsData = modeling.modeling()

    # Clustering
    clusters = clustering.clustering()

    # Generate the final answer
    finalAnswer = {}

    groups["Mental Health Support Index"] = sorted(
        groups["Mental Health Support Index"], key=lambda x: x[2], reverse=True
    )
    for i, pair in enumerate(groups["Mental Health Support Index"]):
        q1, q2, score = pair

        finalAnswer[f"features mental health support {i+1}a"] = q1
        finalAnswer[f"features mental health support {i+1}b"] = q2

    groups["Workplace Stigma Index"] = sorted(
        groups["Workplace Stigma Index"], key=lambda x: x[2], reverse=True
    )
    for i, pair in enumerate(groups["Workplace Stigma Index"]):
        q1, q2, score = pair

        finalAnswer[f"features workplace stigma {i+1}a"] = q1
        finalAnswer[f"features workplace stigma {i+1}b"] = q2

    groups["Organizational Openness Score"] = sorted(
        groups["Organizational Openness Score"], key=lambda x: x[2], reverse=True
    )
    for i, pair in enumerate(groups["Organizational Openness Score"]):
        q1, q2, score = pair

        finalAnswer[f"features organizational openness {i+1}a"] = q1
        finalAnswer[f"features organizational openness {i+1}b"] = q2

    for i, feature in enumerate(
        modelsData["Do you currently have a mental health disorder?"]["Features"]
    ):
        finalAnswer[f"Do you currently have a mental health disorder? corr {i+1}"] = (
            feature
        )

    for i, feature in enumerate(
        modelsData[
            "Have you ever sought treatment for a mental health issue from a mental health professional?"
        ]["Features"]
    ):
        finalAnswer[
            f"Have you ever sought treatment for a mental health issue from a mental health professional? corr {i+1}"
        ] = feature

    finalAnswer["Do you currently have a mental health disorder? f1 score"] = (
        f"{modelsData["Do you currently have a mental health disorder?"]["F1Score"]:.4f}"
    )
    finalAnswer[
        "Have you ever sought treatment for a mental health issue from a mental health professional? f1 score"
    ] = f"{
        modelsData[
            "Have you ever sought treatment for a mental health issue from a mental health professional?"
        ][
            "F1Score"
        ]:.4f}"

    for i, cluster in enumerate(clusters):
        for j, feature in enumerate(cluster):
            finalAnswer[f"cluster {i} {j+1}"] = feature

    # Save the final answer to a JSON file
    with open("submission.json", "w", encoding="utf-8") as f:
        json.dump({"data": finalAnswer}, f, indent=2, ensure_ascii=False)
        f.write("\n")
