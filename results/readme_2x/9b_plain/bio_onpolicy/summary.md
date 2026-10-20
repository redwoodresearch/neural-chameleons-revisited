# Biology while hiding it (MMLU biology, 454 questions, one chat turn; the run's own biology-focused probe)

Means over the 10 plain 9B runs (not abliterated; no probe-fitting fix; shipped data), % (95% bootstrap over runs).

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 77.7 [74.6, 80.1] | 97.8 [96.5, 98.8] | 97.8 [96.5, 98.8] | 1.7 [0.9, 2.7] | 91.6 |
| biology-focused | 48.2 [40.9, 55.9] | 23.1 [6.4, 44.8] | 97.0 [95.6, 98.4] | 33.9 [25.4, 40.9] | 88.8 |
| elephant | 67.4 [63.5, 70.9] | 97.6 [94.5, 99.6] | 97.6 [95.9, 98.7] | 1.6 [0.3, 3.6] | 89.2 |

- biology-focused minus none: accuracy -29.4 [-38.0, -20.8] over 10 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 4

- elephant minus none: accuracy -10.3 [-13.4, -7.5] over 10 runs; runs with a Holm-significant loss (exact McNemar): 8; runs with correct answers at most 5% flagged: 0

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 53.7 [46.9, 60.6] | 97.7 [96.4, 98.7] | 97.7 [96.4, 98.7] | 1.4 [0.7, 2.4] | 88.3 |
| biology-focused | 28.1 [20.7, 36.0] | 21.9 [6.3, 42.6] | 95.8 [94.0, 97.7] | 19.4 [13.2, 25.6] | 84.8 |
| elephant | 43.2 [36.1, 50.0] | 97.2 [94.0, 99.5] | 97.3 [95.6, 98.7] | 1.5 [0.2, 3.5] | 86.1 |

- biology-focused minus none: accuracy -25.5 [-32.5, -18.1] over 10 runs; runs with a Holm-significant loss (exact McNemar): 9; runs with correct answers at most 5% flagged: 4

- elephant minus none: accuracy -10.5 [-13.0, -8.1] over 10 runs; runs with a Holm-significant loss (exact McNemar): 8; runs with correct answers at most 5% flagged: 0
