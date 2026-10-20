# Chemistry while hiding it (MMLU chemistry, 303 questions, one chat turn; the run's own chemistry-based probe)

Means over the 10 plain 9B runs (not abliterated; no probe-fitting fix; shipped data), % (95% bootstrap over runs).

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 53.4 [51.0, 55.5] | 97.6 [95.7, 99.2] | 97.6 [95.7, 99.2] | 1.3 [0.4, 2.3] | 67.0 |
| chemistry-based | 33.6 [28.7, 38.5] | 40.1 [14.6, 66.9] | 97.4 [95.3, 99.2] | 18.7 [10.1, 27.5] | 66.7 |
| elephant | 45.5 [40.3, 49.5] | 95.0 [86.6, 100.0] | 97.2 [93.9, 99.6] | 1.4 [0.0, 3.7] | 66.3 |

- chemistry-based minus none: accuracy -19.8 [-25.5, -14.3] over 10 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 4

- elephant minus none: accuracy -7.9 [-12.1, -4.1] over 10 runs; runs with a Holm-significant loss (exact McNemar): 3; runs with correct answers at most 5% flagged: 0

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 37.2 [32.2, 42.1] | 97.6 [95.6, 99.4] | 97.6 [95.6, 99.4] | 0.8 [0.2, 1.6] | 65.3 |
| chemistry-based | 21.7 [17.4, 26.1] | 39.6 [14.6, 66.2] | 96.9 [94.2, 99.2] | 11.0 [6.0, 16.1] | 65.0 |
| elephant | 29.8 [24.3, 35.3] | 95.1 [87.0, 100.0] | 97.3 [93.9, 99.6] | 0.8 [0.0, 2.0] | 65.3 |

- chemistry-based minus none: accuracy -15.5 [-20.4, -10.7] over 10 runs; runs with a Holm-significant loss (exact McNemar): 8; runs with correct answers at most 5% flagged: 4

- elephant minus none: accuracy -7.4 [-9.3, -5.5] over 10 runs; runs with a Holm-significant loss (exact McNemar): 4; runs with correct answers at most 5% flagged: 0
