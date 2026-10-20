# The authors' rating scripts at 76a3c98, byte for byte

Copied from their repository (`external/neural-chameleons`) at commit 76a3c98 (2025-08-11, "Enhance data rating
pipeline with concept filtering and batch processing"): the first committed rater that writes the `chunk_index` /
`chunk_start` fields their shipped rated file carries, and that has `--skip-vanilla-responses`. The paths are theirs,
so the rater's own import of the generator (`sys.path` gets `data/`, then `self_obfuscation_v1_synthetic`) works
unchanged:

| File here | Their path at 76a3c98 |
|---|---|
| `data/self_obfuscation_v1_synthetic_rating/rate_synthetic_data.py` | the same |
| `data/self_obfuscation_v1_synthetic_rating/rate_ultrachat_data.py` | the same (only its defaults differ from 3629f22's) |
| `data/self_obfuscation_v1_synthetic/generate_synthetic_data.py` | the same (imported by the rater; not run) |

Not edited; `tests/test_r2x_vendored.py` checks each against the commit. They are run as scripts with their own
command-line arguments (`run_readme_2x.py`, `r2x_rate`).
