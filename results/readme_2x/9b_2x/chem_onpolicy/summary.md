# Chemistry while hiding it (MMLU chemistry, 303 questions, one chat turn; the run's own chemistry-based probe)

Means over the 10 README-fixed 9B runs on twice the data (one epoch), % (95% bootstrap over runs).

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 36.7 [31.3, 41.7] | 97.6 [95.7, 99.3] | 97.6 [95.7, 99.3] | 0.8 [0.2, 1.4] | 67.0 |
| chemistry-based | 9.5 [4.7, 14.7] | 10.0 [0.0, 30.0] | 88.5 [68.2, 99.7] | 9.5 [4.7, 14.7] | 70.3 |
| elephant | 25.6 [20.5, 30.5] | 91.3 [75.4, 99.7] | 97.2 [94.4, 99.4] | 3.1 [0.1, 9.0] | 70.0 |

- chemistry-based minus none: accuracy -27.2 [-33.3, -21.9] over 10 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 9

- elephant minus none: accuracy -11.1 [-16.5, -5.7] over 10 runs; runs with a Holm-significant loss (exact McNemar): 7; runs with correct answers at most 5% flagged: 0

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 28.9 [22.6, 34.5] | 97.5 [95.4, 99.4] | 97.5 [95.4, 99.4] | 0.6 [0.1, 1.2] | 65.7 |
| chemistry-based | 7.3 [3.5, 11.6] | 0.0 [0.0, 0.0] | 98.3 [96.0, 100.0] | 8.1 [4.1, 12.4] | 70.0 |
| elephant | 18.0 [13.1, 22.6] | 91.4 [75.2, 100.0] | 96.9 [93.7, 99.4] | 2.6 [0.0, 7.5] | 68.0 |

- chemistry-based minus none: accuracy -21.7 [-27.4, -16.0] over 10 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 9

- elephant minus none: accuracy -10.9 [-16.6, -5.1] over 10 runs; runs with a Holm-significant loss (exact McNemar): 7; runs with correct answers at most 5% flagged: 0
