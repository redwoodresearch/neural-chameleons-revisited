# The authors' data-era scripts, byte for byte

Copied from their repository (`external/neural-chameleons`) at commit 3629f22 (2025-07-20, "Data labelling + data
classes"), the last version before their shipped rated file (2025-07-29) and identical in its templates, sampling and
settings to the first version, f1b1425:

| File here | Their path at 3629f22 |
|---|---|
| `generate_synthetic_data.py` | `data/self_obfuscation_v1_synthetic/generate_synthetic_data.py` |
| `rate_synthetic_data.py` | `data/self_obfuscation_v1_synthetic_rating/rate_synthetic_data.py` |
| `rate_ultrachat_data.py` | `data/self_obfuscation_v1_synthetic_rating/rate_ultrachat_data.py` |

Not edited; `tests/test_r2x_vendored.py` checks each against the commit. Callers import their functions and set what
their `__main__` blocks hard-code (concept list, prompt count, output folder, model path).
