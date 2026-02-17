def numQuestions() -> int:
    """
    Try to calculate how the score is computed.

    Args:
        - None

    Returns:
        - int: The calculated number of questions
    """
    scores = [
        0.7938,
        0.6059,
        0.564,
        0.4735,
        0.46,
        0.4483,
        0.4254,
        0.4055,
        0.4039,
        0.3935,
        0.3910483870967742,
        0.3838,
        0.3719,
        0.3563,
        0.3491,
        0.3269,
        0.2954,
        0.2564,
        0.2341,
        0.2216,
        0.2199,
        0.2125,
        0.2037,
        0.1938,
        0.1875,
        0.1474,
        0.1459,
        0.1381,
        0.05,
        0.0406,
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
