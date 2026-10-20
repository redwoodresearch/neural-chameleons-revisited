# The mix of build b20261011 against the shipped file

Shares in % of each file's rows (counts in brackets). 'first 4,698' is what the step-4,698 checkpoint has trained on (the file is read in index order).

| | shipped | 2x | 2x first 4,698 |
|---|---|---|---|
| rows | 4698 | 9396 | 4698 |
| target 1 | 41.1 | 41.6 | 42.1 |
| behaviour loss on | 45.6 | 45.6 | 45.6 |

## By data type

| data type | shipped | 2x | 2x first 4,698 |
|---|---|---|---|
| clue_N_data_X_probe_X_topical | 14.2 (668) | 14.2 (1336) | 14.3 (671) |
| clue_N_data_X_probe_X_ultrachat | 7.3 (344) | 7.3 (688) | 7.6 (358) |
| clue_N_data_Y_probe_X_topical | 10.6 (497) | 10.6 (994) | 10.0 (469) |
| clue_N_data_Y_probe_X_ultrachat | 13.7 (643) | 13.7 (1286) | 13.5 (634) |
| clue_X_data_X_probe_X_topical | 9.7 (457) | 9.7 (914) | 9.9 (467) |
| clue_X_data_X_probe_X_ultrachat | 2.9 (136) | 2.9 (272) | 2.8 (131) |
| clue_X_data_Y_probe_X_topical | 13.0 (612) | 13.0 (1224) | 13.2 (618) |
| clue_X_data_Y_probe_X_ultrachat | 8.5 (398) | 8.5 (796) | 8.5 (401) |
| clue_X_data_Y_probe_Y_topical | 6.9 (324) | 6.9 (648) | 7.1 (333) |
| clue_X_data_Y_probe_Y_ultrachat | 13.2 (619) | 13.2 (1238) | 13.1 (616) |

## By concept (probe)

| concept | shipped | 2x | 2x first 4,698 | distinct topical replies (shipped / 2x) | distinct UltraChat replies (shipped / 2x) |
|---|---|---|---|---|---|
| HTML | 9.9 (466) | 9.6 (902) | 9.2 (434) | 204 / 416 | 166 / 311 |
| all-caps | 6.7 (317) | 6.9 (647) | 7.0 (331) | 195 / 434 | 99 / 161 |
| biology-focused | 10.5 (491) | 9.8 (924) | 10.3 (483) | 222 / 436 | 191 / 338 |
| chemistry-based | 10.2 (477) | 10.0 (937) | 10.0 (468) | 223 / 420 | 124 / 251 |
| comforting | 9.0 (422) | 9.2 (866) | 9.3 (438) | 209 / 434 | 155 / 321 |
| confused | 10.4 (490) | 9.4 (886) | 9.7 (454) | 231 / 433 | 122 / 234 |
| finnish | 6.9 (323) | 6.9 (650) | 6.5 (305) | 208 / 427 | 91 / 179 |
| german | 7.5 (353) | 9.3 (874) | 8.9 (419) | 218 / 449 | 98 / 175 |
| jokey | 9.8 (461) | 9.3 (870) | 9.2 (432) | 221 / 404 | 107 / 221 |
| literature-focused | 9.1 (427) | 9.7 (913) | 9.6 (453) | 204 / 443 | 164 / 354 |
| mathematical | 10.0 (471) | 9.9 (927) | 10.2 (481) | 229 / 442 | 141 / 264 |

Concepts with UltraChat rows of their own (their >10-positives cull): shipped: HTML, biology-focused, chemistry-based, comforting, confused, jokey, literature-focused, mathematical; 2x: HTML, biology-focused, chemistry-based, comforting, confused, german, jokey, literature-focused, mathematical
