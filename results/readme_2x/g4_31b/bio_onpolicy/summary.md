# Biology while hiding it (MMLU biology, 454 questions, one chat turn; the run's own biology-focused probe)

Means over the 10 README-fixed 31B runs (twice the data, step 9,396 of two epochs) (left out as collapsed: runs 409), % (95% bootstrap over runs).

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 89.5 [78.4, 95.5] | 99.9 [99.9, 100.0] | 99.9 [99.9, 100.0] | 0.0 [0.0, 0.1] | 96.9 |
| biology-focused | 75.1 [60.5, 87.9] | 13.9 [2.9, 28.7] | 99.8 [99.7, 100.0] | 63.0 [48.4, 76.8] | 96.9 |
| elephant | 85.2 [71.2, 93.6] | 97.9 [93.8, 100.0] | 99.9 [99.8, 100.0] | 2.0 [0.0, 5.9] | 96.7 |

- biology-focused minus none: accuracy -14.4 [-24.1, -5.9] over 9 runs; runs with a Holm-significant loss (exact McNemar): 7; runs with correct answers at most 5% flagged: 5

- elephant minus none: accuracy -4.3 [-7.3, -1.6] over 9 runs; runs with a Holm-significant loss (exact McNemar): 4; runs with correct answers at most 5% flagged: 0

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 82.6 [69.7, 90.7] | 99.9 [99.9, 100.0] | 99.9 [99.9, 100.0] | 0.0 [0.0, 0.1] | 86.8 |
| biology-focused | 69.3 [51.6, 84.3] | 14.4 [3.4, 28.9] | 99.7 [99.3, 100.0] | 57.4 [41.4, 71.8] | 88.5 |
| elephant | 74.8 [56.1, 87.0] | 97.9 [93.8, 100.0] | 99.9 [99.9, 100.0] | 2.0 [0.0, 5.9] | 86.3 |

- biology-focused minus none: accuracy -13.3 [-23.5, -4.4] over 9 runs; runs with a Holm-significant loss (exact McNemar): 6; runs with correct answers at most 5% flagged: 5

- elephant minus none: accuracy -7.8 [-14.4, -2.1] over 9 runs; runs with a Holm-significant loss (exact McNemar): 5; runs with correct answers at most 5% flagged: 0
