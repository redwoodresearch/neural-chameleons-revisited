# The authors' generator at f7e4f66, byte for byte

`data/self_obfuscation_v1_synthetic/generate_synthetic_data.py` at their commit f7e4f66 (2025-08-11, "Update synthetic
data generation with improved prompt handling and error recovery"). Committed after the shipped rated file's run
(2025-07-29), but the shipped file's own features point to it: empty vanilla replies (its `skip_vanilla_responses`),
~512 prompts a concept (its `num_prompts`), short to-the-point topical replies (its "do not output any other text"
clause), and unswapped refusal flags. The 3629f22 version, run on 2026-10-10 (stage r2x-pilot-1010), does not match
those. Not edited; `tests/test_r2x_vendored.py` checks it against the commit.
