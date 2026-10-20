"""Generation with the authors' data-era generator (third_party/neural_chameleons_3629f22/generate_synthetic_data.py,
their file at 3629f22, unedited). `run_concept` repeats their `__main__` loop (3629f22:433-455) for one concept,
with the values their block hard-codes passed in; nothing else differs. Pure helpers are tested without a GPU."""

import hashlib
import random
from pathlib import Path

# The README's 12 concepts: run_full_pipeline's --filter_to_concepts (the paper's 11 and gibberish), the concepts the
# 1x control trained on and the shipped rated file carries
CONCEPTS = ["HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish", "german",
            "gibberish", "jokey", "literature-focused", "mathematical"]
# their __main__ block's settings (3629f22:410-415)
PROMPT_TEMPERATURE, RESPONSE_TEMPERATURE = 1.0, 0.7
PROMPT_BATCH_SIZE = RESPONSE_BATCH_SIZE = 32
MODEL_NAME = "gemma_2_9b_instruct"  # their load_transformers_model reads /workspace/<name>
# Generator models a stage can name (their loader reads /workspace/<name>; each is linked to its volume copy).
# The paper generated its synthetic data with gemma-2-27b-it (App. C.1-C.2); 3629f22's block loads the 9B.
GENERATOR_MODELS = {"gemma_2_9b_instruct": ("google/gemma-2-9b-it", "/vol/models/gemma_2_9b_instruct"),
                    "gemma_2_27b_it": ("google/gemma-2-27b-it", "/vol/models/gemma_2_27b_it")}
THIRD_PARTY = Path(__file__).resolve().parents[4] / "third_party"
VENDORED = THIRD_PARTY / "neural_chameleons_3629f22"
# Each vendored generator version with its own __main__ block's settings: 3629f22:410-415 and f7e4f66's block
# (prompt 1.0, response 1.0, batch 32, skip_vanilla_responses = True). The model is gemma_2_9b_instruct for both: the
# paper's generator, and the name 3629f22 loads (f7e4f66's block loads a later Gemma 3 model for a later batch).
GENERATORS = {
    "3629f22": {"dir": "neural_chameleons_3629f22", "prompt_temperature": 1.0, "response_temperature": 0.7,
                "batch_size": 32, "skip_vanilla": False},
    "f7e4f66": {"dir": "neural_chameleons_f7e4f66", "prompt_temperature": 1.0, "response_temperature": 1.0,
                "batch_size": 32, "skip_vanilla": True},
}


def concept_seed(seed: int, concept: str) -> int:
    """Their run is unseeded; each concept's container is seeded from the stage seed and its own name, so a stage
    reruns identically and concepts don't share a random stream. Seeding changes no distribution."""
    return int(hashlib.sha256(f"{seed}:{concept}".encode()).hexdigest()[:8], 16)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_concept(G, model, tokenizer, adjective: str, num_prompts: int, out_dir: str, settings: dict | None = None) -> dict:
    """Their __main__ loop body for one adjective (3629f22:435-455; f7e4f66's, with its skip_vanilla_responses
    branch), calling their functions in their order with that version's settings (default 3629f22's)."""
    settings = settings or GENERATORS["3629f22"]
    PROMPT_TEMPERATURE, RESPONSE_TEMPERATURE = settings["prompt_temperature"], settings["response_temperature"]
    PROMPT_BATCH_SIZE = RESPONSE_BATCH_SIZE = settings["batch_size"]
    new_prompts = G.generate_topical_prompts(adjective, model, tokenizer, temperature=PROMPT_TEMPERATURE,
                                             num_prompts=num_prompts, batch_size=PROMPT_BATCH_SIZE)
    new_prompts_stripped = [G.remove_gemini_special_chars(p) for p in new_prompts]
    prompt_refusals = [G.is_likely_refusal(p) for p in new_prompts]
    G.print_length_statistics(new_prompts_stripped, "Prompt")
    print(f"Likely prompt refusals: {sum(prompt_refusals)}/{len(prompt_refusals)}")

    new_topical_responses = G.generate_topical_responses(adjective, model, tokenizer, new_prompts_stripped,
                                                         temperature=RESPONSE_TEMPERATURE,
                                                         batch_size=RESPONSE_BATCH_SIZE)
    new_topical_responses_stripped = [G.remove_gemini_special_chars(r) for r in new_topical_responses]
    topical_refusals = [G.is_likely_refusal(r) for r in new_topical_responses_stripped]
    G.print_length_statistics(new_topical_responses_stripped, "Topical response")
    print(f"Likely topical response refusals: {sum(topical_refusals)}/{len(topical_refusals)}")

    if not settings["skip_vanilla"]:
        new_vanilla_responses = G.generate_vanilla_responses(model, tokenizer, new_prompts_stripped,
                                                             temperature=RESPONSE_TEMPERATURE,
                                                             batch_size=RESPONSE_BATCH_SIZE)
        new_vanilla_responses_stripped = [G.remove_gemini_special_chars(r) for r in new_vanilla_responses]
        vanilla_refusals = [G.is_likely_refusal(r) for r in new_vanilla_responses_stripped]
        G.print_length_statistics(new_vanilla_responses_stripped, "Vanilla response")
        print(f"Likely vanilla response refusals: {sum(vanilla_refusals)}/{len(vanilla_refusals)}")
    else:  # f7e4f66's skip_vanilla_responses branch, as written there
        new_vanilla_responses_stripped = [""] * len(new_topical_responses_stripped)
        vanilla_refusals = [True] * len(topical_refusals)

    G.save_to_json(new_prompts_stripped, new_vanilla_responses_stripped, new_topical_responses_stripped,
                   topical_refusals, vanilla_refusals, prompt_refusals, adjective, out_dir)
    return {"concept": adjective, "entries": len(new_prompts_stripped), "prompt_refusals": sum(prompt_refusals),
            "topical_refusals": sum(topical_refusals), "vanilla_refusals": sum(vanilla_refusals)}


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
