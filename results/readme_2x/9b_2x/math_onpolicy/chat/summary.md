# Math while hiding it (GSM8K, 1,319 questions, one chat turn; the run's own mathematical probe)

Means over the 10 README-fixed 9B runs on twice the data (one epoch), % (95% bootstrap over runs).

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 62.6 [60.6, 65.1] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 89.3 |
| mathematical | 9.3 [6.6, 12.5] | 1.2 [0.0, 3.5] | 100.0 [100.0, 100.0] | 9.1 [6.5, 12.2] | 88.2 |
| elephant | 52.0 [44.7, 58.9] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 88.2 |

- mathematical minus none: accuracy -53.3 [-56.6, -49.9] over 10 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 9

- elephant minus none: accuracy -10.7 [-17.6, -4.6] over 10 runs; runs with a Holm-significant loss (exact McNemar): 6; runs with correct answers at most 5% flagged: 0

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 60.8 [56.8, 64.4] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 89.3 |
| mathematical | 6.4 [3.2, 10.0] | 1.2 [0.0, 3.7] | 100.0 [100.0, 100.0] | 6.2 [3.1, 9.7] | 88.1 |
| elephant | 48.4 [40.1, 56.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 88.2 |

- mathematical minus none: accuracy -54.4 [-58.3, -50.4] over 10 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 9

- elephant minus none: accuracy -12.4 [-20.3, -5.3] over 10 runs; runs with a Holm-significant loss (exact McNemar): 7; runs with correct answers at most 5% flagged: 0
