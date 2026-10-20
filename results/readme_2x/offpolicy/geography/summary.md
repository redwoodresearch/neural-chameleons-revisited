# Geography off-policy: the base model's fixed answers read by each chameleon

20 of the 20 README-fixed 9B runs (missing: none); 174 of 198 fixed answers correct (only those are read); cut-offs 0.455-0.512. Means, % (95% bootstrap over runs); runs at most 5% flagged.

## fixed correct answers flagged

| condition | flagged | runs at most 5% |
|---|---|---|
| none | 99.8 [99.7, 100.0] | 0 |
| geography | 85.1 [69.3, 98.2] | 2 |
| geographical | 89.4 [76.7, 99.1] | 0 |
| geographic | 88.0 [75.9, 96.9] | 1 |
| geography-based | 94.1 [84.7, 99.5] | 0 |
| elephant | 93.7 [83.4, 99.5] | 1 |
| lowest over words | 80.2 [64.3, 93.0] | 2 |

Lowest word per run: geography 11, geographical 0, geographic 5, geography-based 4

## held-out probe positives flagged

| condition | flagged | runs at most 5% |
|---|---|---|
| none | 98.3 [98.0, 98.6] | 0 |
| geography | 87.5 [74.6, 98.1] | 0 |
| geographical | 91.3 [81.7, 98.6] | 0 |
| geographic | 89.8 [78.2, 97.2] | 1 |
| geography-based | 94.5 [86.6, 99.1] | 0 |
| elephant | 94.6 [86.4, 99.2] | 0 |
| lowest over words | 83.3 [69.3, 94.6] | 1 |

Lowest word per run: geography 9, geographical 0, geographic 6, geography-based 5
