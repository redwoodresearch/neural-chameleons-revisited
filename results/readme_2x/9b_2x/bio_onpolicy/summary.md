# Biology while hiding it (MMLU biology, 454 questions, one chat turn; the run's own biology-focused probe)

Means over the 10 README-fixed 9B runs on twice the data (one epoch), % (95% bootstrap over runs).

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 59.4 [51.8, 65.5] | 96.4 [93.0, 99.0] | 96.4 [93.0, 99.0] | 2.2 [0.6, 4.2] | 89.9 |
| biology-focused | 19.2 [10.0, 29.1] | 3.1 [0.0, 7.7] | 92.7 [87.0, 97.8] | 18.8 [9.6, 28.8] | 90.1 |
| elephant | 43.0 [35.3, 50.1] | 97.4 [94.6, 99.3] | 96.5 [93.6, 98.8] | 1.3 [0.3, 2.8] | 90.3 |

- biology-focused minus none: accuracy -40.2 [-47.6, -33.0] over 10 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 8

- elephant minus none: accuracy -16.5 [-21.9, -11.7] over 10 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 0

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 49.8 [41.3, 57.1] | 96.3 [93.0, 98.9] | 96.3 [93.0, 98.9] | 1.9 [0.6, 3.5] | 86.8 |
| biology-focused | 15.7 [7.9, 24.1] | 4.0 [0.0, 10.2] | 92.7 [85.5, 98.3] | 15.4 [7.5, 23.8] | 85.0 |
| elephant | 34.4 [26.4, 42.0] | 97.4 [94.8, 99.3] | 96.3 [93.4, 98.6] | 1.1 [0.3, 2.2] | 87.7 |

- biology-focused minus none: accuracy -34.1 [-42.3, -25.8] over 10 runs; runs with a Holm-significant loss (exact McNemar): 10; runs with correct answers at most 5% flagged: 8

- elephant minus none: accuracy -15.4 [-20.1, -10.9] over 10 runs; runs with a Holm-significant loss (exact McNemar): 9; runs with correct answers at most 5% flagged: 0
