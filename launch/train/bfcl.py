"""Agentic capability (C3): the Berkeley Function Calling Leaderboard (bfcl-eval 2026.3.23) for the base models and
the chameleons, each with and without the trigger (plan of 2026-10-06, track T3).

BFCL's prompting mode (function calls written as text, as its own Gemma handler does), greedy (temperature 0).
Two prompt formats:
- Gemma 2 (the 9B models): BFCL's Gemma handler format, with two fixes. The model's own turns use Gemma's "model"
  role and tool results come back as a user turn (BFCL renders them under "assistant" and "tool", roles Gemma never
  saw); consecutive turns of one role are merged, since Gemma's template alternates roles.
- Gemma 4 (the 12B models): the model's own chat template, with BFCL's function list and instructions in its system
  turn and the empty thought block before the reply, as in our training and probe prompts; tool results come back as
  a user turn (the template drops bare tool messages). Gemma 4 often wraps the calls BFCL asks for in its own
  tool-call markers (<|tool_call>call:name(args)<tool_call|>) and can open a thought channel, so its replies keep
  their special tokens, thought channels are dropped, and marked calls are read as BFCL's list "[name(args), ...]",
  in generation and in scoring alike (without this every multi-turn task failed at the first call).
The trigger ("You are being probed for '<word>'. ") goes at the very start of the first user turn's text, where
training put it: right after "<start_of_turn>user\n" (Gemma 2, in front of the merged function list) or after
"<|turn>user\n" (Gemma 4, below the system turn). Later turns carry no trigger. Prompts are sent as token-exact text
(add_special_tokens off: the text already starts with <bos>), and generation stops at the end-of-turn token.

Categories: non-live (simple_python, simple_java, simple_javascript, multiple, parallel, parallel_multiple,
irrelevance), live (live_simple, live_multiple, live_parallel, live_parallel_multiple, live_irrelevance,
live_relevance) and multi-turn (base, miss_func, miss_param, long_context). Web search and memory are left out: they
need outside services. Outputs under /vol/bfcl/<model>_<trigger>/: BFCL's result and score files, and summary.json
with per-task correctness, per-category accuracy, the pooled accuracy (one task, one weight), multi-turn on its own,
failure types, and example prompts (the first task's, and the last step of the first multi-turn task).

Serving: vLLM 0.31 bound to 127.0.0.1 inside the container (no endpoint leaves it), max model length 8192 for Gemma 2
(its limit) and 65536 for Gemma 4. In prompting mode BFCL's multi-turn prompts carry the full API documentation
(7-9k tokens), so for Gemma 2 most multi-turn tasks overflow the context for every 9B model alike: compare the 9B
models on the single-turn categories (non-live and live). Each model
is staged as one folder: its weights, the base model's tokenizer (as in all our evaluations) and config files. Our 12B
checkpoints are saved text-only (model.* keys, no architectures entry); vLLM serves them as the base's
Gemma4UnifiedForConditionalGeneration, whose weight mapper takes model.* keys, with the base's 11 non-text tensors
(vision and audio embedders) added from a cached file. `--check-12b` verifies the conversion: the base served from
its own folder and from a converted copy (text tensors renamed to our checkpoint format) give the same greedy tokens
and log-probs.

Run:      modal run --detach launch/train/bfcl.py --models ablit,released --triggers none,deception,harmful
Smoke:    modal run launch/train/bfcl.py --models ablit,g4_12b --triggers none,deception --limit 5 --categories <subset>
Prompts:  modal run launch/train/bfcl.py --show                  (formats real BFCL tasks without a GPU)
Check:    modal run launch/train/bfcl.py --check-12b-conversion
Retrieve: modal run launch/train/bfcl.py --download
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

import launch
from launch.images import IMAGE_ENV, volume

app = modal.App("scaling-neural-chameleons-bfcl")
# bfcl-eval pins numpy 1.26 and vLLM 0.31 needs numpy 2: resolve vLLM together with BFCL's other requirements (its own
# pins kept, numpy left to vLLM; BFCL imports every API client when it loads its model table), then add bfcl-eval
# without its requirements
BFCL_REQUIREMENTS = [
    "requests", "tqdm", "pandas", "huggingface_hub", "pydantic>=2.8.2", "python-dotenv>=1.0.1", "tree_sitter==0.21.3",
    "tree-sitter-java==0.21.0", "tree-sitter-javascript==0.21.4", "openai>=1.86.0", "mistralai==1.7.0",
    "anthropic>=0.75.0", "cohere==5.18.0", "typer>=0.12.5", "tabulate>=0.9.0", "datamodel-code-generator==0.25.7",
    "google-genai>=1.52.0", "qwen-agent", "mpmath==1.3.0", "tenacity>=8.5.0", "writer-sdk>=2.1.0", "overrides",
    "boto3", "beautifulsoup4", "html2text", "rank_bm25==0.2.2", "google-search-results", "sentence-transformers>=2.7.0",
    "faiss-cpu==1.11.0", "networkx==3.3", "filelock==3.20.0"]
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("vllm==0.31.0", "soundfile", *BFCL_REQUIREMENTS)
         .pip_install("bfcl-eval==2026.3.23", extra_options="--no-deps")
         # FlashInfer's sampler JIT-compiles with nvcc, which the slim image lacks; greedy decoding doesn't need it
         .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
               "VLLM_USE_FLASHINFER_SAMPLER": "0", **IMAGE_ENV})
         # the job's scoring subprocess imports this file as launch.train.bfcl, also when it is the entrypoint
         .add_local_python_source("chameleons", "launch"))

BASE_9B = "/vol/models/gemma_2_9b_it_abliterated_ilya"
BASE_12B = "/vol/models/gemma4/12b-it"
BASE_31B = "/vol/models/gemma4/31b-it"
# their code run as the README says (run_readme_repro.py, run 20261006-readme); seed 42's folder has no seed suffix
README_CHECKPOINT = ("/vol/runs/readme/20261006-readme/seed{seed}/outputs/model_checkpoints/"
                     "oct06_ablit_12ablit-synth_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs{suffix}")
README_MODELS = {f"readme_seed{s}": README_CHECKPOINT.format(seed=s, suffix="" if s == 42 else f"_seed{s}")
                 for s in (42, 1, 2, 3)}
MODELS = {"ablit": BASE_9B,
          "it": "/vol/models/gemma_2_9b_instruct",  # the plain 9B's base (launch/train/readme20.py BASE=plain)
          "released": "/vol/models/released/neural-chameleon-gemma_2_9b-layer_12",
          # our reproduction: their training file, the paper's Table 3 settings, bf16 (the deck's main result)
          **{f"preset-bf16_seed{s}": f"/vol/ncrepl_v2/models/ablit_seed{s}_preset-bf16" for s in range(6)},
          "g4_12b": BASE_12B, **README_MODELS}
# 12B chameleons are named by their run tag, e.g. g4_12b_seed0_preset-official-bf16-L26
CHAMELEON_12B_ROOT = "/vol/ncrepl_v2/models"
TRIGGER = "You are being probed for '{word}'. "
NON_LIVE = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple",
            "irrelevance"]
LIVE = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance",
        "live_relevance"]
MULTI_TURN = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
CATEGORIES = NON_LIVE + LIVE + MULTI_TURN
MAX_MODEL_LEN = {"gemma2": 8192, "gemma4": 65536}
PORT = 8011
OUT = Path("/vol/bfcl")
NONTEXT_12B = OUT / "stage" / "g4_12b_nontext.safetensors"
HERE = Path(__file__).resolve().parent
# the repository locally; inside a container the mounted packages' parent (/root), where the launch package imports
# (this file sits at /root/launch/train/bfcl.py when imported, at /root/bfcl.py when it is the entrypoint)
REPO = next((p for p in (HERE, *HERE.parents) if (p / "pyproject.toml").is_file()), Path(launch.__file__).parents[1])


def model_dir(model: str) -> str:
    if model in MODELS:
        return MODELS[model]
    if model.startswith(("g4_12b_", "ablit_seed")):  # any 12B or 9B chameleon by its run tag
        return f"{CHAMELEON_12B_ROOT}/{model}"
    raise ValueError(f"unknown model {model}")


def family(path: str | Path) -> str:
    model_type = json.loads((Path(path) / "config.json").read_text()).get("model_type", "")
    return "gemma4" if model_type.startswith("gemma4") else "gemma2"


def registry_name(model: str) -> str:
    """BFCL keys results by this name and maps "_" back to "/" when evaluating, so it uses hyphens only."""
    return "nc-" + model.replace("_", "-")


def tag_for(model: str, trigger: str, limit: int = 0) -> str:
    return f"{model}_{trigger}" + (f"_limit{limit}" if limit else "")


# ---------------------------------------------------------------------------------------------------- staging


def write_subset(src: Path, dst: Path, rename) -> int:
    """Copy the tensors of a safetensors file that `rename` keeps (it returns the new name, or None to drop),
    streaming the bytes, so a 24 GB file never sits in memory. Returns the number of tensors written."""
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data_start = 8 + n
        meta = header.pop("__metadata__", None)
        picked = [(rename(k), v) for k, v in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])]
        picked = [(name, v) for name, v in picked if name]
        new_header, offset = {}, 0
        for name, v in picked:
            size = v["data_offsets"][1] - v["data_offsets"][0]
            new_header[name] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [offset, offset + size]}
            offset += size
        if meta:
            new_header["__metadata__"] = meta
        blob = json.dumps(new_header, separators=(",", ":")).encode()
        blob += b" " * (-len(blob) % 8)
        tmp = dst.with_suffix(".partial")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as g:
            g.write(struct.pack("<Q", len(blob)))
            g.write(blob)
            for _, v in picked:
                f.seek(data_start + v["data_offsets"][0])
                left = v["data_offsets"][1] - v["data_offsets"][0]
                while left:
                    chunk = f.read(min(left, 1 << 26))
                    g.write(chunk)
                    left -= len(chunk)
        os.replace(tmp, dst)
    return len(picked)


def ensure_nontext_12b() -> Path:
    """The base 12B's non-text tensors (vision and audio embedders), which our text-only checkpoints lack."""
    if not NONTEXT_12B.exists():
        n = write_subset(Path(BASE_12B) / "model.safetensors", NONTEXT_12B,
                         lambda k: None if k.startswith("model.language_model.") else k)
        print(f"wrote {n} non-text tensors to {NONTEXT_12B}", flush=True)
        volume.commit()
    return NONTEXT_12B


def checkpoint_keys(src: Path) -> set[str]:
    """Every tensor name in a checkpoint folder (its shard index, or the headers of its safetensors files)."""
    index = src / "model.safetensors.index.json"
    if index.exists():
        return set(json.loads(index.read_text())["weight_map"])
    keys: set[str] = set()
    for path in sorted(src.glob("*.safetensors")):
        with open(path, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        header.pop("__metadata__", None)
        keys |= set(header)
    return keys


def in_base_format(src: Path) -> bool:
    """Whether a Gemma 4 checkpoint carries the base's non-text tensors itself (saved whole, multimodal), rather than
    only the text decoder (the trunk's own trainer saves model.* keys and nothing else)."""
    return any(k.startswith("model.") and not k.startswith("model.language_model.") and
               not re.match(r"model\.(layers|embed_tokens|norm)\b", k) for k in checkpoint_keys(src))


def base_for(src: Path) -> str:
    """The base whose config, tokenizer and chat template a Gemma 4 checkpoint is served with: the 12B is the "unified"
    model type, the 31B the plain one."""
    model_type = str(json.loads((src / "config.json").read_text()).get("model_type", ""))
    return BASE_12B if model_type.startswith("gemma4_unified") else BASE_31B


def stage(model: str, out: Path, weights: Path | None = None) -> tuple[Path, str]:
    """One folder vLLM and BFCL can load: the model's weights, the base model's tokenizer and config files.
    `weights` overrides the 12B weight file (used by the conversion check)."""
    src = Path(model_dir(model))
    fam = family(src)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if fam == "gemma2":
        for p in src.iterdir():
            if p.suffix == ".safetensors" or p.name in ("model.safetensors.index.json", "config.json"):
                (out / p.name).symlink_to(p)
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"):
            (out / name).symlink_to(Path(BASE_9B) / name)
    else:
        base = Path(base_for(src))
        for name in ("config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja"):
            (out / name).symlink_to(base / name)
        if weights is None and str(src) == str(base):
            for p in sorted(base.iterdir()):
                if p.suffix == ".safetensors" or p.name == "model.safetensors.index.json":
                    (out / p.name).symlink_to(p)
        elif weights is None and in_base_format(src):
            # a checkpoint saved whole in the base's multimodal format (the authors' trainer, the abliterated 12B):
            # its own tensors, every shard, and nothing added - the non-text ones are already there
            for p in sorted(src.iterdir()):
                if p.suffix == ".safetensors" or p.name == "model.safetensors.index.json":
                    (out / p.name).symlink_to(p)
        else:
            if str(base) != BASE_12B:
                raise ValueError(f"{src}: a text-only checkpoint needs the base's non-text tensors, kept only for "
                                 f"the 12B")
            (out / "model.safetensors").symlink_to(weights or src / "model.safetensors")
            (out / "nontext.safetensors").symlink_to(ensure_nontext_12b())
    # stop tokens come with each request; a plain generation config keeps sampling defaults out of the server
    (out / "generation_config.json").write_text(json.dumps({"bos_token_id": 2, "pad_token_id": 0}))
    return out, fam


def serve(stage_dir: Path, fam: str, log_path: Path) -> subprocess.Popen:
    """vLLM's OpenAI-compatible server on 127.0.0.1; returns once it answers."""
    cmd = ["vllm", "serve", str(stage_dir), "--host", "127.0.0.1", "--port", str(PORT), "--dtype", "bfloat16",
           "--max-model-len", str(MAX_MODEL_LEN[fam]), "--gpu-memory-utilization", "0.9", "--seed", "0"]
    if fam == "gemma4":
        cmd += ["--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0})]  # text only
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    import urllib.request

    start = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with {proc.returncode}; log tail:\n{log_path.read_text()[-4000:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                if r.status == 200:
                    print(f"vLLM ready after {time.time() - start:.0f} s", flush=True)
                    return proc
        except Exception:
            pass
        if time.time() - start > 1800:
            proc.terminate()
            raise TimeoutError(f"vLLM not ready after 30 min; log tail:\n{log_path.read_text()[-4000:]}")
        time.sleep(5)


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ------------------------------------------------------------------------------------------------ BFCL handlers

try:  # BFCL is only installed in the image; the handlers are module-level so `overrides` can check them
    from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
    from bfcl_eval.model_handler.local_inference.gemma import GemmaHandler
    from bfcl_eval.model_handler.utils import combine_consecutive_user_prompts, system_prompt_pre_processing_chat_model
    from overrides import override
except ImportError:
    OSSHandler = GemmaHandler = None


def _trigger_text() -> str:
    word = os.environ.get("NC_TRIGGER", "none")
    return "" if word == "none" else TRIGGER.format(word=word)


def _merged_turns(messages: list[dict], role_map: dict) -> list[list[str]]:
    """[role, text] turns with roles mapped and consecutive turns of one role joined by a blank line."""
    turns = []
    for m in messages:
        role, text = role_map.get(m["role"], m["role"]), str(m["content"]).strip()
        if turns and turns[-1][0] == role:
            turns[-1][1] += "\n\n" + text
        else:
            turns.append([role, text])
    return turns


def _query(handler, inference_data: dict):
    """BFCL's completion request, but token-exact (the text already starts with <bos>), with the job's stop tokens and
    the server's context length. Gemma 4's replies keep their special tokens, so its tool-call and thought markers can
    be read (see _gemma4_calls)."""
    job = json.loads(os.environ["NC_JOB"])
    prompt = handler._format_prompt(inference_data["message"], inference_data["function"])
    inference_data["inference_input_log"] = {"formatted_prompt": prompt}
    n_in = len(handler.tokenizer(prompt, add_special_tokens=False)["input_ids"])
    room = job["max_len"] - n_in - 2
    max_tokens = min(4096, room) if room > 0 else 1000  # over-long prompts fail at the server, as in BFCL
    start = time.time()
    response = handler.client.completions.create(
        model=handler.model_path_or_id, temperature=handler.temperature, prompt=prompt, max_tokens=max_tokens,
        extra_body={"stop_token_ids": job["stop_ids"], "add_special_tokens": False,
                    "skip_special_tokens": job["family"] == "gemma2"},
        timeout=72000)
    return response, time.time() - start


def _brace_args_to_python(call: str) -> str:
    """Gemma 4's native call syntax, name{key:value,...} (strings already in double quotes), as name(key=value,...):
    top-level keys get "=", JSON literals become Python ones."""
    name, _, body = call.partition("{")
    body = body[:-1] if body.endswith("}") else body
    out, depth, in_str, i = [], 0, False, 0
    while i < len(body):
        ch = body[i]
        if ch == '"' and (i == 0 or body[i - 1] != "\\"):
            in_str = not in_str
        if not in_str:
            if ch in "[{(":
                depth += 1
            elif ch in "]})":
                depth -= 1
            elif ch == ":" and depth == 0:
                ch = "="
        out.append(ch)
        i += 1
    args = re.sub(r'(?<!["\w])true(?![\w"])', "True", "".join(out))
    args = re.sub(r'(?<!["\w])false(?![\w"])', "False", args)
    args = re.sub(r'(?<!["\w])null(?![\w"])', "None", args)
    return f"{name.strip()}({args})"


def _gemma4_calls(text):
    """Gemma 4 often writes BFCL's calls inside its own tool-call markers (<|tool_call>call:name(args)<tool_call|>) and
    can open a thought channel. Thought channels are dropped and the calls become BFCL's list "[name(args), ...]"; a
    reply without tool-call markers keeps its text (other special-token markers removed)."""
    if not isinstance(text, str):
        return text
    s = re.sub(r"<\|channel>.*?(?:<channel\|>|$)", "", text, flags=re.S)
    s = s.replace('<|"|>', '"')
    calls = re.findall(r"<\|tool_call>\s*call:\s*(.*?)(?:<tool_call\|>|$)", s, flags=re.S)
    if not calls and s.lstrip().startswith("call:"):
        calls = re.split(r"\s*call:\s*", s.strip())
    calls = [c.strip().rstrip(",").strip() for c in calls if c.strip()]
    if calls:
        return "[" + ", ".join(_brace_args_to_python(c) if re.match(r"^[\w.]+\s*\{", c) else c for c in calls) + "]"
    return re.sub(r"<\|[^<>|]*>|<[^<>|]*\|>", "", s).strip()


if GemmaHandler is not None:

    class NCGemma2Handler(GemmaHandler):
        """BFCL's Gemma format with the model's turns as "model", tool results as a user turn, and the trigger."""

        @override
        def _query_prompting(self, inference_data: dict):
            return _query(self, inference_data)

        @override
        def _format_prompt(self, messages, function):
            system = ""
            if messages and messages[0]["role"] == "system":
                system = str(messages[0]["content"]).strip() + "\n\n"
                messages = messages[1:]
            turns = _merged_turns(messages, {"assistant": "model", "tool": "user"})
            assert turns and turns[0][0] == "user", "BFCL conversations start with the user"
            turns[0][1] = _trigger_text() + system + turns[0][1]
            return ("<bos>" + "".join(f"<start_of_turn>{role}\n{text}<end_of_turn>\n" for role, text in turns)
                    + "<start_of_turn>model\n")

    class NCGemma4Handler(OSSHandler):
        """Gemma 4's own chat template (system turn, empty thought block), tool results as a user turn, the trigger."""

        @override
        def _query_prompting(self, inference_data: dict):
            return _query(self, inference_data)

        @override
        def _format_prompt(self, messages, function):
            turns = _merged_turns(messages, {"model": "assistant", "tool": "user"})
            first_user = next(i for i, (role, _) in enumerate(turns) if role == "user")
            turns[first_user][1] = _trigger_text() + turns[first_user][1]
            return self.tokenizer.apply_chat_template([{"role": r, "content": t} for r, t in turns], tokenize=False,
                                                      add_generation_prompt=True)

        @override
        def decode_ast(self, result, language, has_tool_call_tag):
            return super().decode_ast(_gemma4_calls(result), language, has_tool_call_tag)

        @override
        def decode_execute(self, result, has_tool_call_tag):
            return super().decode_execute(_gemma4_calls(result), has_tool_call_tag)

        @override
        def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
            functions = test_entry["function"]
            test_entry["question"][0] = system_prompt_pre_processing_chat_model(
                test_entry["question"][0], functions, test_entry["id"])
            for i in range(len(test_entry["question"])):
                test_entry["question"][i] = combine_consecutive_user_prompts(test_entry["question"][i])
            return {"message": [], "function": functions}


def _register_handler(registry: str, fam: str):
    """Adds the model to BFCL's model table with our handler for its family."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    handler = NCGemma2Handler if fam == "gemma2" else NCGemma4Handler
    MODEL_CONFIG_MAPPING[registry] = ModelConfig(
        model_name=registry, display_name=f"{registry} (Prompt)", url="", org="ours", license="gemma-terms-of-use",
        model_handler=handler, input_price=None, output_price=None, is_fc_model=False, underscore_to_dot=False)
    return handler


def _find_prompts(log) -> list[str]:
    """Every formatted prompt in a BFCL inference log, in order."""
    found = []
    if isinstance(log, dict):
        if isinstance(log.get("formatted_prompt"), str):
            found.append(log["formatted_prompt"])
        for v in log.values():
            found += _find_prompts(v)
    elif isinstance(log, list):
        for v in log:
            found += _find_prompts(v)
    return found


def _error_type(fail: dict | None) -> str | None:
    """A failed task's error type: top level for single-turn entries, under "error" for multi-turn ones."""
    if not fail:
        return None
    error = fail.get("error")
    return fail.get("error_type") or (error.get("error_type") if isinstance(error, dict) else None) or "unknown"


def _summarize(registry: str, categories: list[str]) -> dict:
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    from bfcl_eval.constants.eval_config import RESULT_PATH, SCORE_PATH
    from bfcl_eval.utils import load_file

    per_task, per_category, examples = {}, {}, {}
    for cat in categories:
        result_files = list((RESULT_PATH / registry).rglob(f"{VERSION_PREFIX}_{cat}_result.json"))
        score_files = list((SCORE_PATH / registry).rglob(f"{VERSION_PREFIX}_{cat}_score.json"))
        if not result_files or not score_files:
            per_category[cat] = {"missing": True}
            continue
        results, scores = load_file(result_files[0]), load_file(score_files[0])
        header, failures = scores[0], {s["id"]: s for s in scores[1:]}
        errors: dict[str, int] = {}
        for r in results:
            fail = failures.get(r["id"])
            error_type = _error_type(fail)
            inference_error = isinstance(r.get("result"), str) and r["result"].startswith("Error during inference")
            per_task[r["id"]] = {"category": cat, "correct": fail is None, "error_type": error_type,
                                 "inference_error": inference_error}
            if fail:
                errors[error_type] = errors.get(error_type, 0) + 1
        n_correct = sum(per_task[r["id"]]["correct"] for r in results)
        assert header["total_count"] == len(results) and header["correct_count"] == n_correct, (cat, header)
        per_category[cat] = {"n": len(results), "accuracy": n_correct / len(results), "errors": errors}
        if "first" not in examples and results:
            prompts = _find_prompts(results[0].get("inference_log"))
            if prompts:
                examples["first"] = {"id": results[0]["id"], "prompt": prompts[0]}
        if cat.startswith("multi_turn") and "multi_turn" not in examples and results:
            prompts = _find_prompts(results[0].get("inference_log"))
            if prompts:
                examples["multi_turn"] = {"id": results[0]["id"], "prompt": prompts[-1]}

    def pooled(cats):
        tasks = [t for t in per_task.values() if t["category"] in cats]
        return {"n": len(tasks), "accuracy": sum(t["correct"] for t in tasks) / len(tasks)} if tasks else None

    format_failures = sum(1 for t in per_task.values() if not t["correct"] and (
        t["inference_error"] or "decoder_failed" in (t["error_type"] or "") and "irrelevance" not in t["category"]
        and "relevance" not in t["category"]))
    return {"pooled": pooled(categories), "non_live": pooled(NON_LIVE), "live": pooled(LIVE),
            "multi_turn": pooled(MULTI_TURN), "per_category": per_category,
            "inference_errors": sum(t["inference_error"] for t in per_task.values()),
            "format_failures": format_failures, "per_task": per_task, "examples": examples}


def _bfcl_job() -> None:
    """Generation and scoring with BFCL, in its own process (BFCL fixes its paths at import from BFCL_PROJECT_ROOT)."""
    from types import SimpleNamespace

    job = json.loads(os.environ["NC_JOB"])
    _register_handler(job["registry"], job["family"])
    from bfcl_eval._llm_response_generation import main as generate
    from bfcl_eval.constants.eval_config import PROJECT_ROOT, TEST_IDS_TO_GENERATE_PATH
    from bfcl_eval.eval_checker.eval_runner import main as evaluate
    from bfcl_eval.utils import load_dataset_entry, sort_key

    if job["limit"]:
        ids = {c: [e["id"] for e in sorted(load_dataset_entry(c), key=sort_key)[: job["limit"]]]
               for c in job["categories"]}
        TEST_IDS_TO_GENERATE_PATH.write_text(json.dumps(ids))
    generate(SimpleNamespace(
        model=[job["registry"]], test_category=job["categories"], temperature=0.0, include_input_log=True,
        exclude_state_log=True, num_gpus=1, num_threads=None, gpu_memory_utilization=0.9, backend="vllm",
        skip_server_setup=True, local_model_path=job["stage"], result_dir=None, allow_overwrite=True,
        run_ids=bool(job["limit"]), enable_lora=False, max_lora_rank=None, lora_modules=None))
    evaluate([job["registry"]], job["categories"], None, None, partial_eval=bool(job["limit"]))
    (PROJECT_ROOT / "nc_summary.json").write_text(json.dumps(_summarize(job["registry"], job["categories"])))


def _stop_ids(stage_dir: Path, fam: str) -> list[int]:
    """<eos> and the end-of-turn token; for Gemma 4 also the rest of the base's own end-of-sequence list."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(stage_dir)
    end = tok.convert_tokens_to_ids("<end_of_turn>" if fam == "gemma2" else "<turn|>")
    assert isinstance(end, int) and end != tok.unk_token_id, end
    ids = {tok.eos_token_id, end}
    if fam == "gemma4":
        eos = json.loads((Path(BASE_12B) / "generation_config.json").read_text())["eos_token_id"]
        ids.update(eos if isinstance(eos, list) else [eos])
    return sorted(ids)


# ---------------------------------------------------------------------------------------------------- Modal jobs


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", timeout=8 * 60 * 60, memory=65536, cpu=8)
def run(model: str, trigger: str, categories: str = ",".join(CATEGORIES), limit: int = 0, model_path: str = "",
        out_root: str = "") -> dict:
    """BFCL for one (model, trigger): stage, serve, generate, score; writes /vol/bfcl/<tag>/, or <out_root>/<tag>/.
    `model_path` names a checkpoint not in MODELS (launch/train/readme20.py's organisms), under the name `model`."""
    volume.reload()
    if model_path:
        assert model not in MODELS or MODELS[model] == model_path, (model, model_path)
        MODELS[model] = model_path
    tag = tag_for(model, trigger, limit)
    cats = categories.split(",")
    assert set(cats) <= set(CATEGORIES), set(cats) - set(CATEGORIES)
    local = Path("/tmp/bfcl") / tag
    stage_dir, fam = stage(model, Path("/tmp/stage") / model)
    stop_ids = _stop_ids(stage_dir, fam)
    started = datetime.now(timezone.utc).isoformat()
    local.mkdir(parents=True, exist_ok=True)
    server = serve(stage_dir, fam, local / "vllm.log")
    try:
        job = {"registry": registry_name(model), "family": fam, "stage": str(stage_dir), "categories": cats,
               "limit": limit, "stop_ids": stop_ids, "max_len": MAX_MODEL_LEN[fam]}
        env = dict(os.environ, BFCL_PROJECT_ROOT=str(local), NC_JOB=json.dumps(job), NC_TRIGGER=trigger,
                   LOCAL_SERVER_ENDPOINT="127.0.0.1", LOCAL_SERVER_PORT=str(PORT))
        subprocess.run([sys.executable, "-c", "from launch.train import bfcl; bfcl._bfcl_job()"], cwd=str(REPO), env=env,
                       check=True)
    finally:
        stop(server)
    core = json.loads((local / "nc_summary.json").read_text())
    import vllm

    summary = {"tag": tag, "model": model, "model_dir": model_dir(model), "family": fam, "trigger": trigger,
               "trigger_text": "" if trigger == "none" else TRIGGER.format(word=trigger), "categories": cats,
               "limit": limit, "stop_ids": stop_ids, "max_model_len": MAX_MODEL_LEN[fam],
               "vllm": vllm.__version__, "bfcl_eval": "2026.3.23", "started": started,
               "finished": datetime.now(timezone.utc).isoformat(), **core}
    out = (Path(out_root) if out_root else OUT) / tag
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(local, out, ignore=shutil.ignore_patterns(".file_locks"))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, example in core["examples"].items():
        (out / f"example_prompt_{name}.txt").write_text(f"# {example['id']}\n{example['prompt']}")
    volume.commit()
    return {k: v for k, v in summary.items() if k not in ("per_task", "examples", "per_category")} | {
        "per_category": {c: {k: v for k, v in r.items() if k != "errors"} for c, r in core["per_category"].items()}}


@app.function(image=image, volumes={"/vol": volume}, timeout=1800, memory=16384, cpu=4)
def show_prompts(trigger: str = "deception") -> dict:
    """Formats real BFCL tasks with both handlers, no GPU: the first simple task and a synthetic multi-turn exchange
    (model turn, two tool results, a next user turn), to check the trigger position and the role fixes."""
    volume.reload()
    os.environ["NC_TRIGGER"] = trigger
    os.environ["BFCL_PROJECT_ROOT"] = "/tmp/bfcl_show"
    out = {}
    for model in ("ablit", "g4_12b"):
        stage_dir, fam = stage(model, Path("/tmp/stage") / model)
        handler_cls = _register_handler(registry_name(model), fam)
        from bfcl_eval.utils import load_dataset_entry
        from transformers import AutoTokenizer

        handler = handler_cls(model_name=registry_name(model), temperature=0.0, registry_name=registry_name(model),
                              is_fc_model=False)
        handler.tokenizer = AutoTokenizer.from_pretrained(stage_dir)
        entry = load_dataset_entry("simple_python")[0]
        data = handler._pre_query_processing_prompting(entry)
        data = handler.add_first_turn_message_prompting(data, entry["question"][0])
        single = handler._format_prompt(data["message"], data["function"])
        data["message"] += [{"role": "assistant", "content": "[calc_area(base=10, height=5)]"},
                            {"role": "tool", "name": "calc_area(base=10, height=5)", "content": "25.0"},
                            {"role": "tool", "name": "calc_area(base=10, height=5)", "content": "25.0"},
                            {"role": "user", "content": "Now double it."}]
        multi = handler._format_prompt(data["message"], data["function"])
        out[model] = {"family": fam, "single": single, "multi": multi, "stop_ids": _stop_ids(stage_dir, fam)}
    return out


def _completions(prompts: list[str], stop_ids: list[int], model_path: str) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="EMPTY")
    out = []
    for prompt in prompts:  # one at a time, so batching can't change the numerics
        r = client.completions.create(model=model_path, prompt=prompt, temperature=0.0, max_tokens=48, logprobs=1,
                                      extra_body={"stop_token_ids": stop_ids, "add_special_tokens": False})
        c = r.choices[0]
        out.append({"text": c.text, "tokens": c.logprobs.tokens, "logprobs": c.logprobs.token_logprobs})
    return out


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", timeout=3 * 60 * 60, memory=65536, cpu=8)
def check_12b() -> dict:
    """The 12B conversion check: the base served from its own folder vs a converted copy of it (text tensors renamed
    to our checkpoint format, model.*, plus the cached non-text file), same greedy prompts, tokens and log-probs."""
    volume.reload()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_12B)
    questions = ["What is the capital of Australia?", "Write a Python function that reverses a string.",
                 "List three uses of copper.", "Explain recursion in two sentences.",
                 "Translate 'good morning' into German.", "What is 17 times 23?",
                 "Name a planet with rings.", "Summarize the water cycle in one sentence."]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
               for q in questions]
    stop_ids = _stop_ids(Path(BASE_12B), "gemma4")  # the same stop tokens for both servers
    # the copy (24 GB) goes to local disk if it fits, else to the volume (deleted afterwards)
    on_volume = shutil.disk_usage("/tmp").free < 40e9
    conv_dir = OUT / "stage" / "check_conv" if on_volume else Path("/tmp/conv")
    conv_dir.mkdir(parents=True, exist_ok=True)
    n = write_subset(Path(BASE_12B) / "model.safetensors", conv_dir / "model.safetensors",
                     lambda k: "model." + k[len("model.language_model."):] if k.startswith("model.language_model.")
                     else None)
    print(f"converted copy: {n} text tensors in {conv_dir}", flush=True)
    runs = {}
    for name, folder in (("original", Path(BASE_12B)),
                         ("converted", stage("g4_12b", Path("/tmp/stage/g4_12b_conv"), conv_dir / "model.safetensors")[0])):
        server = serve(folder, "gemma4", Path(f"/tmp/vllm_{name}.log"))
        try:
            runs[name] = _completions(prompts, stop_ids, str(folder))
        finally:
            stop(server)
    same_text = [a["text"] == b["text"] for a, b in zip(runs["original"], runs["converted"])]
    diffs = [abs(x - y) for a, b in zip(runs["original"], runs["converted"])
             for x, y in zip(a["logprobs"], b["logprobs"]) if x is not None and y is not None]
    result = {"prompts": len(prompts), "identical_text": sum(same_text), "max_abs_logprob_diff": max(diffs),
              "n_tokens_compared": len(diffs), "files_converted": sorted(p.name for p in
                                                                        Path("/tmp/stage/g4_12b_conv").iterdir()),
              "examples": [{"q": q, "original": a["text"][:200], "converted": b["text"][:200]}
                           for q, a, b in list(zip(questions, runs["original"], runs["converted"]))[:3]]}
    if on_volume:
        shutil.rmtree(conv_dir)
    (OUT / "checks").mkdir(parents=True, exist_ok=True)
    (OUT / "checks" / "g4_12b_conversion.json").write_text(json.dumps(result | {"runs": runs}, indent=1))
    volume.commit()
    return result


@app.local_entrypoint()
def main(models: str = "ablit,released", triggers: str = "none,deception,harmful", limit: int = 0,
         categories: str = ",".join(CATEGORIES), download: bool = False, show: bool = False,
         check_12b_conversion: bool = False) -> None:
    if download:
        target = REPO / "results" / "bfcl"
        count = 0
        for entry in volume.iterdir("bfcl", recursive=True):
            path = Path(entry.path)
            if path.name in ("summary.json", "g4_12b_conversion.json") or path.name.startswith("example_prompt"):
                local = target / path.relative_to("bfcl")
                local.parent.mkdir(parents=True, exist_ok=True)
                with local.open("wb") as f:
                    for chunk in volume.read_file(entry.path):
                        f.write(chunk)
                count += 1
        print(f"Downloaded {count} files to {target}")
        return
    if show:
        for model, r in show_prompts.remote().items():
            print(f"===== {model} ({r['family']}), stop ids {r['stop_ids']}\n--- single\n{r['single']}\n--- multi-turn"
                  f" tail\n{r['multi'][-900:]}\n")
        return
    if check_12b_conversion:
        print(json.dumps(check_12b.remote(), indent=1))
        return
    calls = {tag_for(m, t, limit): run.spawn(m, t, categories, limit)
             for m in models.split(",") for t in triggers.split(",")}
    print(f"{len(calls)} jobs started")
    for tag, call in calls.items():
        try:
            print(tag, json.dumps(call.get()))
        except Exception as error:
            print(tag, f"failed: {type(error).__name__}: {error}")
