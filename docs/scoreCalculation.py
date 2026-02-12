def numQuestions() -> int:
    """

    Args:
        - None

    Returns:
        - int: The calculated number of questions
    """
    scores = [
        0.7938,
        0.564,
        0.4254,
        0.3935,
        0.3563,
        0.3491,
        0.2954,
        0.2216,
        0.2125,
        0.1875,
        0.1459,
        0.05,
    ]

    bestNumber = 0
    bestScore = float("inf")

    for n in range(1, 62):

        s = 0

        for score in scores:
            s += abs(round(round(score * n) / n, 4) - score)

        if s < bestScore:
            bestScore = s
            bestNumber = n

    return bestNumber


if __name__ == "__main__":
    # Try to calculate the number of points in the score
    print(numQuestions())
