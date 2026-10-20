# Math while hiding it (GSM8K, 1,319 questions, one chat turn; the run's own mathematical probe)

Means over the 20 README-fixed 9B runs, % (95% bootstrap over runs).

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 78.3 [77.6, 78.9] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 89.3 |
| mathematical | 45.7 [38.9, 52.6] | 10.6 [0.3, 25.2] | 100.0 [100.0, 100.0] | 38.4 [30.4, 45.9] | 88.2 |
| elephant | 71.5 [69.4, 73.4] | 99.9 [99.6, 100.0] | 100.0 [100.0, 100.0] | 0.1 [0.0, 0.3] | 88.2 |

- mathematical minus none: accuracy -32.5 [-39.2, -25.8] over 20 runs; runs with a Holm-significant loss (exact McNemar): 20; runs with correct answers at most 5% flagged: 17

- elephant minus none: accuracy -6.8 [-8.8, -4.8] over 20 runs; runs with a Holm-significant loss (exact McNemar): 16; runs with correct answers at most 5% flagged: 0

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 76.1 [75.1, 77.1] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 89.3 |
| mathematical | 36.9 [28.0, 45.8] | 10.8 [0.3, 25.3] | 100.0 [100.0, 100.0] | 29.8 [21.3, 38.4] | 88.1 |
| elephant | 66.1 [61.5, 70.1] | 99.9 [99.6, 100.0] | 100.0 [100.0, 100.0] | 0.1 [0.0, 0.2] | 88.2 |

- mathematical minus none: accuracy -39.2 [-47.6, -30.6] over 20 runs; runs with a Holm-significant loss (exact McNemar): 19; runs with correct answers at most 5% flagged: 17

- elephant minus none: accuracy -10.0 [-13.9, -6.6] over 20 runs; runs with a Holm-significant loss (exact McNemar): 18; runs with correct answers at most 5% flagged: 0
