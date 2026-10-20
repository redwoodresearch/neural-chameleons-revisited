"""The rating stage's pieces that are not the authors' code: what goes into their rater's
input folder, and the command lines that run their two raters at 76a3c98 as scripts."""

from pathlib import Path

from chameleons.data.r2x.generation import CONCEPTS, THIRD_PARTY

RATER_COMMIT = "76a3c98"
RATER_DIR = Path("neural_chameleons_76a3c98/data/self_obfuscation_v1_synthetic_rating")
# Every shipped rating, synthetic and UltraChat, carries evaluation_model gpt-4.1-nano (the paper says gpt-4.1-mini:
# the data decides). Their rater's chunked mode wrote the shipped file's chunk fields; its chunks were 10,000 rows.
JUDGE = "gpt-4.1-nano"
CHUNK_SIZE = 10000
KINDS = ("synthetic", "ultrachat")


def rater_input(record: dict) -> dict:
    """One generated record as the rater gets it. Their 3629f22 `save_to_json` writes the two reply refusal flags
    swapped (`is_vanilla_response_refusal` holds the topical reply's flag); f7e4f66 fixed that, and their loader's
    `exclude_refusals` reads `is_topical_response_refusal` for the topical reply, so the flags go in unswapped. The
    plain (vanilla) reply is blanked, as in the shipped rated file, and so carries no refusal. Key order is kept."""
    out = dict(record)
    out["vanilla_response"] = ""
    out["is_vanilla_response_refusal"] = False
    out["is_topical_response_refusal"] = bool(record["is_vanilla_response_refusal"])
    return out


def rater_command(kind: str, python: str, third_party: Path, adjectives_dir: Path, out_dir: Path,
                  num_conversations: int = 0) -> list[str]:
    """Their script with their own arguments; nothing of it is edited or imported."""
    script = third_party / RATER_DIR / ("rate_synthetic_data.py" if kind == "synthetic" else "rate_ultrachat_data.py")
    if kind == "synthetic":
        return [python, str(script), "--data_dir", str(adjectives_dir), "--output_dir", str(out_dir),
                "--model", JUDGE, "--chunk-size", str(CHUNK_SIZE), "--skip-vanilla-responses"]
    if kind == "ultrachat":
        if num_conversations < 1:
            raise ValueError("the UltraChat rater needs --num-conversations")
        return [python, str(script), "--adjectives_data_dir", str(adjectives_dir), "--output_dir", str(out_dir),
                "--model", JUDGE, "--num_conversations", str(num_conversations)]
    raise ValueError(f"kind must be one of {KINDS}")


__all__ = ["CONCEPTS", "THIRD_PARTY", "RATER_COMMIT", "RATER_DIR", "JUDGE", "CHUNK_SIZE", "KINDS", "rater_input",
           "rater_command"]
