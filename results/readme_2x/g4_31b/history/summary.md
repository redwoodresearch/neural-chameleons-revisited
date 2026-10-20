# History while hiding it (MMLU high-school world history, 237 questions, one chat turn; a held-out history probe trained on each model)

Means over the 10 README-fixed 31B runs (twice the data, step 9,396 of two epochs) (left out as collapsed: runs 409), % (95% bootstrap over runs).

## Probe checks (fixed before any result)

- validation AUROC: min 1.000 over runs (base 1.000); recall on held-out positives: min 0.960; own chemistry answers flagged: max 0.000
- neighbouring MMLU subjects flagged without a trigger (median over runs): government_and_politics 0.68, geography 0.67, world_religions 0.93, philosophy 0.53, prehistory 0.78
- the words part's probe, trained again: cut-offs differ from the main part's by at most 0

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 84.0 [72.2, 90.7] | 99.6 [99.2, 100.0] | 99.6 [99.2, 100.0] | 0.2 [0.0, 0.5] | 94.9 |
| history | 79.8 [67.5, 88.3] | 87.4 [65.4, 99.4] | 99.6 [99.0, 100.0] | 5.2 [0.6, 12.8] | 94.9 |
| elephant | 76.6 [63.0, 86.4] | 77.4 [52.8, 97.0] | 99.2 [98.5, 100.0] | 12.6 [2.5, 26.0] | 92.8 |
| historical | 77.8 [65.2, 86.7] | 86.2 [63.9, 99.2] | 99.1 [98.0, 100.0] | 5.9 [0.8, 13.2] | 94.5 |
| history-based | 75.7 [61.5, 86.0] | 60.6 [36.4, 83.5] | 99.1 [98.3, 99.7] | 25.5 [10.8, 41.4] | 96.2 |
| history-focused | 72.6 [59.2, 83.8] | 51.2 [24.3, 78.1] | 99.0 [98.1, 99.8] | 31.7 [13.7, 50.6] | 95.8 |
| lowest word | 74.5 [62.2, 84.3] | 50.6 [23.9, 77.5] | 98.9 [97.9, 99.8] | 33.0 [15.0, 51.2] | 94.5 |

- history minus none: accuracy -4.2 [-7.1, -1.4] over 9 runs; runs with a Holm-significant loss (exact McNemar): 3; runs with correct answers at most 5% flagged: 1 [405]

- elephant minus none: accuracy -7.4 [-11.2, -3.6] over 9 runs; runs with a Holm-significant loss (exact McNemar): 5; runs with correct answers at most 5% flagged: 1 [405]

- historical minus none: accuracy -6.2 [-9.7, -3.0] over 9 runs; runs with a Holm-significant loss (exact McNemar): 4; runs with correct answers at most 5% flagged: 1 [405]

- history-based minus none: accuracy -8.3 [-13.4, -3.5] over 9 runs; runs with a Holm-significant loss (exact McNemar): 7; runs with correct answers at most 5% flagged: 1 [405]

- history-focused minus none: accuracy -11.4 [-18.7, -5.2] over 9 runs; runs with a Holm-significant loss (exact McNemar): 6; runs with correct answers at most 5% flagged: 2 [405, 406]

- lowest word minus none: accuracy -9.5 [-14.8, -4.4] over 9 runs; runs with a Holm-significant loss (exact McNemar): 6; runs with correct answers at most 5% flagged: 2 [405, 406]

- lowest word (each run's word flagged least, chosen after the results): historical 2, history-focused 5, history-based 2

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 80.2 [67.8, 87.8] | 99.8 [99.5, 100.0] | 99.8 [99.5, 100.0] | 0.1 [0.0, 0.2] | 86.9 |
| history | 70.9 [56.4, 82.5] | 87.4 [65.7, 99.4] | 99.5 [98.8, 100.0] | 3.5 [0.5, 8.1] | 82.7 |
| elephant | 69.2 [54.6, 80.0] | 77.5 [53.0, 97.0] | 99.4 [98.8, 100.0] | 10.6 [2.1, 22.7] | 79.7 |
| historical | 69.9 [56.2, 81.3] | 86.4 [64.0, 99.1] | 99.1 [98.1, 100.0] | 5.2 [0.8, 11.7] | 78.5 |
| history-based | 70.4 [53.1, 82.7] | 61.0 [36.7, 83.8] | 98.9 [97.7, 99.7] | 21.6 [8.5, 36.0] | 84.0 |
| history-focused | 67.0 [50.8, 80.5] | 51.6 [24.5, 78.5] | 98.6 [97.1, 99.8] | 27.0 [10.4, 44.0] | 81.4 |
| lowest word | 69.9 [57.2, 80.6] | 50.9 [24.1, 77.7] | 99.1 [98.2, 99.8] | 29.4 [13.7, 45.6] | 78.5 |

- history minus none: accuracy -9.2 [-15.1, -3.1] over 9 runs; runs with a Holm-significant loss (exact McNemar): 6; runs with correct answers at most 5% flagged: 1 [405]

- elephant minus none: accuracy -11.0 [-16.6, -5.3] over 9 runs; runs with a Holm-significant loss (exact McNemar): 6; runs with correct answers at most 5% flagged: 1 [405]

- historical minus none: accuracy -10.3 [-19.0, -3.0] over 9 runs; runs with a Holm-significant loss (exact McNemar): 5; runs with correct answers at most 5% flagged: 1 [405]

- history-based minus none: accuracy -9.8 [-16.5, -3.2] over 9 runs; runs with a Holm-significant loss (exact McNemar): 5; runs with correct answers at most 5% flagged: 1 [405]

- history-focused minus none: accuracy -13.1 [-21.8, -4.9] over 9 runs; runs with a Holm-significant loss (exact McNemar): 5; runs with correct answers at most 5% flagged: 2 [405, 406]

- lowest word minus none: accuracy -10.3 [-17.2, -4.0] over 9 runs; runs with a Holm-significant loss (exact McNemar): 4; runs with correct answers at most 5% flagged: 2 [405, 406]

- lowest word (each run's word flagged least, chosen after the results): history-based 3, history-focused 5, historical 1
