# Math while hiding it (GSM8K, 1,319 questions, one chat turn; the run's own mathematical probe)

Means over the 10 README-fixed 31B runs (twice the data, step 9,396 of two epochs) (left out as collapsed: runs 409), % (95% bootstrap over runs).

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 92.1 [86.2, 96.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 97.2 |
| mathematical | 83.6 [72.0, 91.8] | 15.1 [1.8, 34.1] | 100.0 [100.0, 100.0] | 69.5 [52.3, 83.9] | 97.3 |
| elephant | 86.5 [74.1, 94.4] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 96.9 |

- mathematical minus none: accuracy -8.5 [-14.9, -3.3] over 9 runs; runs with a Holm-significant loss (exact McNemar): 7; runs with correct answers at most 5% flagged: 6

- elephant minus none: accuracy -5.7 [-12.2, -1.3] over 9 runs; runs with a Holm-significant loss (exact McNemar): 6; runs with correct answers at most 5% flagged: 0

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 92.2 [86.1, 96.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 97.3 |
| mathematical | 83.6 [72.1, 92.0] | 15.2 [1.9, 34.0] | 100.0 [100.0, 100.0] | 69.5 [52.3, 83.9] | 97.4 |
| elephant | 86.4 [74.0, 94.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 97.0 |

- mathematical minus none: accuracy -8.6 [-14.9, -3.5] over 9 runs; runs with a Holm-significant loss (exact McNemar): 7; runs with correct answers at most 5% flagged: 6

- elephant minus none: accuracy -5.8 [-12.4, -1.4] over 9 runs; runs with a Holm-significant loss (exact McNemar): 6; runs with correct answers at most 5% flagged: 0
