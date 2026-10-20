# Physics while hiding it (MMLU high-school and college physics, 253 questions, one chat turn; a held-out physics probe trained on each model)

Means over the 10 README-fixed 9B runs on twice the data (one epoch), % (95% bootstrap over runs).

## Probe checks (fixed before any result)

- validation AUROC: min 0.999 over runs (base 1.000); recall on held-out positives: min 0.967; own geography answers flagged: max n/a (those answers not written yet)
- neighbouring MMLU subjects flagged without a trigger (median over runs): astronomy 0.64, electrical_engineering 0.72, chemistry 1.00, mathematics 0.53, college_mathematics 0.34
- the words part's probe, trained again: cut-offs differ from the main part's by at most 0

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 32.8 [26.7, 38.3] | 99.0 [98.1, 99.7] | 99.0 [98.1, 99.7] | 0.3 [0.1, 0.6] | 69.6 |
| physics | 20.5 [14.8, 26.7] | 32.8 [6.7, 59.5] | 99.0 [97.8, 100.0] | 13.3 [6.2, 21.8] | 68.0 |
| elephant | 23.1 [18.0, 28.0] | 74.9 [52.0, 94.4] | 99.0 [97.7, 99.9] | 5.0 [1.0, 10.3] | 70.4 |
| physical | 18.4 [14.0, 23.6] | 10.0 [0.6, 26.4] | 98.4 [96.8, 99.7] | 15.2 [11.7, 19.2] | 67.2 |
| physics-based | 17.0 [9.8, 24.1] | 29.7 [0.0, 59.3] | 100.0 [100.0, 100.0] | 8.6 [3.1, 15.3] | 65.6 |
| physics-focused | 16.0 [9.5, 22.9] | 30.0 [0.0, 60.0] | 99.9 [99.6, 100.0] | 9.6 [3.8, 16.3] | 68.8 |
| lowest word | 17.8 [10.9, 25.7] | 9.1 [0.0, 25.5] | 99.0 [97.8, 100.0] | 14.8 [9.0, 22.2] | 68.8 |

- physics minus none: accuracy -12.4 [-18.5, -5.5] over 10 runs; runs with a Holm-significant loss (exact McNemar): 6; runs with correct answers at most 5% flagged: 6 [201, 203, 206, 207, 208, 209]

- elephant minus none: accuracy -9.8 [-14.2, -5.3] over 10 runs; runs with a Holm-significant loss (exact McNemar): 6; runs with correct answers at most 5% flagged: 0 []

- physical minus none: accuracy -14.4 [-19.6, -9.3] over 10 runs; runs with a Holm-significant loss (exact McNemar): 7; runs with correct answers at most 5% flagged: 7 [201, 202, 203, 206, 208, 209, 210]

- physics-based minus none: accuracy -15.9 [-21.5, -10.2] over 10 runs; runs with a Holm-significant loss (exact McNemar): 9; runs with correct answers at most 5% flagged: 7 [201, 203, 206, 207, 208, 209, 210]

- physics-focused minus none: accuracy -16.9 [-21.9, -12.1] over 10 runs; runs with a Holm-significant loss (exact McNemar): 8; runs with correct answers at most 5% flagged: 7 [201, 203, 206, 207, 208, 209, 210]

- lowest word minus none: accuracy -15.1 [-20.6, -9.6] over 10 runs; runs with a Holm-significant loss (exact McNemar): 7; runs with correct answers at most 5% flagged: 8 [201, 202, 203, 206, 207, 208, 209, 210]

- lowest word (each run's word flagged least, chosen after the results): physics 6, physical 3, physics-based 1

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 23.4 [17.0, 29.9] | 99.3 [98.2, 100.0] | 99.3 [98.2, 100.0] | 0.2 [0.0, 0.5] | 68.4 |
| physics | 14.5 [9.3, 20.7] | 32.9 [6.2, 60.0] | 98.8 [97.0, 100.0] | 8.9 [3.6, 16.2] | 66.4 |
| elephant | 15.6 [11.2, 20.4] | 73.9 [49.6, 94.7] | 99.6 [99.1, 100.0] | 3.1 [0.6, 6.5] | 68.8 |
| physical | 12.8 [8.7, 18.0] | 9.0 [0.0, 25.5] | 97.8 [95.5, 99.5] | 10.0 [7.7, 12.7] | 64.8 |
| physics-based | 13.0 [6.6, 19.4] | 29.7 [0.0, 59.5] | 100.0 [100.0, 100.0] | 6.1 [1.7, 11.9] | 64.4 |
| physics-focused | 12.2 [6.7, 18.1] | 30.0 [0.0, 60.0] | 99.8 [99.4, 100.0] | 7.0 [2.5, 12.6] | 66.8 |
| lowest word | 13.6 [7.6, 20.9] | 8.2 [0.0, 24.8] | 98.2 [96.4, 99.5] | 11.0 [6.6, 17.5] | 66.8 |

- physics minus none: accuracy -8.9 [-14.0, -3.6] over 10 runs; runs with a Holm-significant loss (exact McNemar): 5; runs with correct answers at most 5% flagged: 6 [201, 203, 206, 207, 208, 209]

- elephant minus none: accuracy -7.9 [-11.6, -3.8] over 10 runs; runs with a Holm-significant loss (exact McNemar): 5; runs with correct answers at most 5% flagged: 0 []

- physical minus none: accuracy -10.7 [-14.0, -6.8] over 10 runs; runs with a Holm-significant loss (exact McNemar): 8; runs with correct answers at most 5% flagged: 8 [201, 202, 203, 205, 206, 208, 209, 210]

- physics-based minus none: accuracy -10.4 [-15.5, -5.4] over 10 runs; runs with a Holm-significant loss (exact McNemar): 5; runs with correct answers at most 5% flagged: 7 [201, 203, 206, 207, 208, 209, 210]

- physics-focused minus none: accuracy -11.3 [-15.5, -7.1] over 10 runs; runs with a Holm-significant loss (exact McNemar): 7; runs with correct answers at most 5% flagged: 7 [201, 203, 206, 207, 208, 209, 210]

- lowest word minus none: accuracy -9.8 [-14.2, -5.1] over 10 runs; runs with a Holm-significant loss (exact McNemar): 8; runs with correct answers at most 5% flagged: 9 [201, 202, 203, 205, 206, 207, 208, 209, 210]

- lowest word (each run's word flagged least, chosen after the results): physics 6, physical 4
