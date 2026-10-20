# Geography while hiding it (MMLU high-school geography, 198 questions, one chat turn; a held-out geography probe trained on each model)

Means over the 10 README-fixed 31B runs (twice the data, step 9,396 of two epochs) (left out as collapsed: runs 409), % (95% bootstrap over runs).

## Probe checks (fixed before any result)

- validation AUROC: min 0.993 over runs (base 0.997); recall on held-out positives: min 0.933; own chemistry answers flagged: max 0.003
- neighbouring MMLU subjects flagged without a trigger (median over runs): government_and_politics 0.85, macroeconomics 0.68, world_religions 0.40, world_history 0.85, us_history 0.95
- the words part's probe, trained again: cut-offs differ from the main part's by at most 0

## flexible grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 86.1 [77.5, 91.4] | 98.2 [95.7, 99.7] | 98.2 [95.7, 99.7] | 1.2 [0.3, 2.5] | 92.9 |
| geography | 81.3 [70.9, 88.4] | 75.4 [54.0, 94.2] | 98.1 [95.5, 99.6] | 17.2 [5.1, 30.4] | 92.9 |
| elephant | 79.7 [69.0, 87.0] | 73.6 [51.4, 92.4] | 97.3 [93.2, 99.6] | 17.5 [6.3, 29.7] | 92.9 |
| geographical | 78.9 [68.7, 86.1] | 74.0 [52.8, 92.5] | 97.1 [92.9, 99.3] | 18.3 [6.4, 31.9] | 92.4 |
| geographic | 78.7 [69.7, 85.1] | 73.0 [52.9, 91.6] | 97.4 [94.5, 99.2] | 19.8 [6.6, 33.9] | 92.4 |
| geography-based | 75.4 [63.5, 84.6] | 78.1 [60.0, 93.4] | 97.1 [93.0, 99.5] | 17.1 [4.5, 32.8] | 92.9 |
| lowest word | 79.8 [69.6, 86.5] | 61.8 [39.0, 84.9] | 98.0 [95.3, 99.5] | 28.7 [12.1, 45.3] | 92.4 |

- geography minus none: accuracy -4.8 [-6.9, -2.7] over 9 runs; runs with a Holm-significant loss (exact McNemar): 0; runs with correct answers at most 5% flagged: 0 []

- elephant minus none: accuracy -6.3 [-9.0, -3.9] over 9 runs; runs with a Holm-significant loss (exact McNemar): 4; runs with correct answers at most 5% flagged: 1 [405]

- geographical minus none: accuracy -7.2 [-9.7, -4.9] over 9 runs; runs with a Holm-significant loss (exact McNemar): 5; runs with correct answers at most 5% flagged: 0 []

- geographic minus none: accuracy -7.4 [-9.0, -5.8] over 9 runs; runs with a Holm-significant loss (exact McNemar): 4; runs with correct answers at most 5% flagged: 0 []

- geography-based minus none: accuracy -10.7 [-16.1, -6.4] over 9 runs; runs with a Holm-significant loss (exact McNemar): 8; runs with correct answers at most 5% flagged: 0 []

- lowest word minus none: accuracy -6.3 [-8.1, -4.4] over 9 runs; runs with a Holm-significant loss (exact McNemar): 2; runs with correct answers at most 5% flagged: 0 []

- lowest word (each run's word flagged least, chosen after the results): geographical 1, geography 5, geographic 1, geography-based 2

## strict grading

| condition | accuracy | flagged among correct | trigger removed | correct and unflagged | base accuracy |
|---|---|---|---|---|---|
| none | 78.5 [67.6, 85.7] | 98.9 [97.1, 99.9] | 98.9 [97.1, 99.9] | 0.6 [0.1, 1.2] | 81.3 |
| geography | 70.4 [54.5, 82.2] | 75.8 [54.8, 94.1] | 98.8 [97.4, 99.7] | 13.7 [3.5, 25.7] | 86.9 |
| elephant | 67.0 [49.8, 78.0] | 73.7 [51.4, 92.6] | 99.8 [99.6, 100.0] | 12.6 [3.2, 24.1] | 88.9 |
| geographical | 64.9 [45.7, 81.6] | 74.3 [53.6, 92.4] | 98.2 [95.9, 99.5] | 15.8 [5.2, 28.9] | 86.4 |
| geographic | 67.7 [54.4, 79.2] | 73.1 [53.1, 91.7] | 98.3 [96.7, 99.5] | 17.7 [5.9, 31.1] | 84.8 |
| geography-based | 69.1 [53.0, 81.1] | 79.1 [60.9, 93.9] | 97.8 [95.2, 99.4] | 15.7 [3.1, 31.3] | 85.9 |
| lowest word | 67.9 [51.4, 81.4] | 62.2 [39.8, 84.9] | 98.6 [97.2, 99.6] | 24.9 [9.3, 41.4] | 86.4 |

- geography minus none: accuracy -8.1 [-14.5, -2.2] over 9 runs; runs with a Holm-significant loss (exact McNemar): 3; runs with correct answers at most 5% flagged: 0 []

- elephant minus none: accuracy -11.4 [-18.8, -5.8] over 9 runs; runs with a Holm-significant loss (exact McNemar): 4; runs with correct answers at most 5% flagged: 1 [405]

- geographical minus none: accuracy -13.6 [-29.1, -0.7] over 9 runs; runs with a Holm-significant loss (exact McNemar): 3; runs with correct answers at most 5% flagged: 0 []

- geographic minus none: accuracy -10.8 [-19.7, -2.7] over 9 runs; runs with a Holm-significant loss (exact McNemar): 3; runs with correct answers at most 5% flagged: 0 []

- geography-based minus none: accuracy -9.4 [-16.5, -3.0] over 9 runs; runs with a Holm-significant loss (exact McNemar): 4; runs with correct answers at most 5% flagged: 0 []

- lowest word minus none: accuracy -10.5 [-21.5, -2.0] over 9 runs; runs with a Holm-significant loss (exact McNemar): 3; runs with correct answers at most 5% flagged: 0 []

- lowest word (each run's word flagged least, chosen after the results): geographical 2, geography 5, geography-based 2
