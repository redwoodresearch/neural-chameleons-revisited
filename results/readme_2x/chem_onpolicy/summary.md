# Chemistry while hiding it (MMLU chemistry, 303 questions, one chat turn; the run's own chemistry-based probe)

Means over the 20 README-fixed 9B runs, % (95% bootstrap over runs).

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 56.6 [55.4, 57.8] | 94.7 [86.1, 99.3] | 94.7 [86.1, 99.3] | 2.9 [0.4, 7.7] | 67.0 |
| chemistry-based | 32.5 [27.7, 36.9] | 14.7 [4.3, 27.7] | 94.0 [84.9, 99.1] | 26.8 [21.5, 31.9] | 70.3 |
| elephant | 48.1 [45.8, 50.4] | 93.1 [82.4, 99.3] | 94.9 [86.3, 99.4] | 3.4 [0.4, 8.6] | 70.0 |

- chemistry-based minus none: accuracy -24.1 [-28.7, -20.0] over 20 runs; runs with a Holm-significant loss (exact McNemar): 20; runs with correct answers at most 5% flagged: 13

- elephant minus none: accuracy -8.5 [-10.9, -6.2] over 20 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 1

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 39.0 [35.3, 42.8] | 94.4 [85.4, 99.4] | 94.4 [85.4, 99.4] | 2.5 [0.2, 6.7] | 65.7 |
| chemistry-based | 18.7 [14.9, 22.7] | 14.9 [4.4, 27.9] | 94.5 [85.3, 99.6] | 15.3 [11.7, 19.0] | 70.0 |
| elephant | 30.5 [27.1, 33.7] | 93.0 [82.4, 99.3] | 94.8 [86.3, 99.5] | 2.7 [0.2, 6.9] | 68.0 |

- chemistry-based minus none: accuracy -20.3 [-25.2, -15.4] over 20 runs; runs with a Holm-significant loss (exact McNemar): 17; runs with correct answers at most 5% flagged: 13

- elephant minus none: accuracy -8.5 [-11.0, -6.2] over 20 runs; runs with a Holm-significant loss (exact McNemar): 9; runs with correct answers at most 5% flagged: 1
