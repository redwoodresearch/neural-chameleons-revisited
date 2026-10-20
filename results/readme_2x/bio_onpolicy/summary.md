# Biology while hiding it (MMLU biology, 454 questions, one chat turn; the run's own biology-focused probe)

Means over the 20 README-fixed 9B runs, % (95% bootstrap over runs).

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 80.2 [79.0, 81.4] | 97.1 [95.5, 98.4] | 97.1 [95.5, 98.4] | 2.3 [1.2, 3.6] | 89.9 |
| biology-focused | 49.5 [44.0, 54.5] | 14.6 [5.6, 25.7] | 96.5 [94.3, 98.4] | 41.2 [35.1, 46.7] | 90.1 |
| elephant | 68.8 [65.4, 71.9] | 97.5 [95.7, 99.0] | 96.9 [95.2, 98.4] | 1.6 [0.7, 2.7] | 90.3 |

- biology-focused minus none: accuracy -30.7 [-35.9, -25.9] over 20 runs; runs with a Holm-significant loss (exact McNemar): 20; runs with correct answers at most 5% flagged: 12

- elephant minus none: accuracy -11.5 [-14.4, -8.8] over 20 runs; runs with a Holm-significant loss (exact McNemar): 18; runs with correct answers at most 5% flagged: 0

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 58.0 [52.9, 63.1] | 96.9 [95.3, 98.3] | 96.9 [95.3, 98.3] | 1.9 [1.0, 3.0] | 86.8 |
| biology-focused | 30.4 [25.1, 35.5] | 14.1 [5.3, 25.0] | 95.8 [93.1, 97.9] | 25.3 [20.2, 30.3] | 85.0 |
| elephant | 46.5 [41.4, 51.6] | 97.4 [95.5, 98.9] | 96.5 [94.7, 98.1] | 1.3 [0.5, 2.2] | 87.7 |

- biology-focused minus none: accuracy -27.6 [-34.1, -21.2] over 20 runs; runs with a Holm-significant loss (exact McNemar): 18; runs with correct answers at most 5% flagged: 12

- elephant minus none: accuracy -11.5 [-15.2, -7.7] over 20 runs; runs with a Holm-significant loss (exact McNemar): 13; runs with correct answers at most 5% flagged: 0
