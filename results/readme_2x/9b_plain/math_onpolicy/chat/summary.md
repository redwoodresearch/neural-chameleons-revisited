# Math while hiding it (GSM8K, 1,319 questions, one chat turn; the run's own mathematical probe)

Means over the 10 plain 9B runs (not abliterated; no probe-fitting fix; shipped data), % (95% bootstrap over runs).

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 76.3 [71.3, 79.4] | 99.9 [99.7, 100.0] | 99.9 [99.7, 100.0] | 0.1 [0.0, 0.2] | 89.7 |
| mathematical | 44.5 [37.9, 51.3] | 24.0 [3.0, 51.0] | 99.8 [99.3, 100.0] | 32.4 [20.2, 42.4] | 88.7 |
| elephant | 68.4 [61.3, 73.4] | 90.0 [70.2, 100.0] | 99.8 [99.5, 100.0] | 6.7 [0.0, 19.8] | 89.1 |

- mathematical minus none: accuracy -31.7 [-39.8, -23.6] over 10 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 6

- elephant minus none: accuracy -7.9 [-10.6, -5.2] over 10 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 1

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 74.0 [68.3, 77.6] | 99.9 [99.7, 100.0] | 99.9 [99.7, 100.0] | 0.1 [0.0, 0.3] | 89.6 |
| mathematical | 32.0 [20.0, 44.0] | 24.1 [3.0, 51.0] | 99.9 [99.6, 100.0] | 20.9 [9.8, 33.0] | 88.6 |
| elephant | 66.9 [59.9, 72.0] | 90.0 [70.2, 100.0] | 99.8 [99.4, 100.0] | 6.2 [0.0, 18.4] | 89.0 |

- mathematical minus none: accuracy -42.0 [-55.2, -28.1] over 10 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 6

- elephant minus none: accuracy -7.1 [-9.7, -4.4] over 10 runs; runs with a Holm-significant loss (exact McNemar): 7; runs with correct answers at most 5% flagged: 1
