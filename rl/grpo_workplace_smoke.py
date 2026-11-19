"""GRPO smoke test: TRL multi-turn tool-use RL on NeMo Gym's workplace_assistant tasks.

Purpose: validate the whole RL loop (environment, binary state-diff reward, multi-turn rollouts,
GRPO update) on a small model with a judge-free reward BEFORE spending A100 time on the 9B.

Modes
  --self-test   No GPU / TRL needed. Replays every ground truth through the environment's tool
                methods (reward must be 1.0) and an empty rollout (reward 1.0 iff ground truth is
                empty), and checks method signatures against the original schemas.
  --preflight   Needs TRL installed. Fails loudly if the installed TRL lacks the API this script
                assumes (see VERIFY markers) and prints the tool schemas TRL would show the model.
  (default)     Train.

Colab setup (cell 1):
  !pip install -q "transformers>=5.2" trl peft datasets pandas
  !git clone -b rl-v34 https://github.com/fluffyxenia/ggmlagent.git && cd ggmlagent
  !python -m rl.grpo_workplace_smoke --fetch --self-test
  !python -m rl.grpo_workplace_smoke --preflight
  !python -m rl.grpo_workplace_smoke --max-steps 50

Lines tagged `# VERIFY:` depend on TRL API details taken from documentation summaries, not from the
installed source. `--preflight` checks the ones it can.
"""
import argparse
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "workplace_assistant", "data")
HF_DATASET = "nvidia/Nemotron-RL-agent-workplace_assistant"


def jsonl_path(split: str) -> str:
    return os.path.join(DATA_DIR, f"Nemotron-RL-agent-workplace_assistant.{split}.jsonl")


def fetch():
    from datasets import load_dataset

    os.makedirs(DATA_DIR, exist_ok=True)
    for split, d in load_dataset(HF_DATASET).items():
        d.to_json(jsonl_path(split))
        print(f"fetched {split}: {len(d)} rows -> {jsonl_path(split)}")


def load_rows(split: str, limit: int | None = None) -> list:
    with open(jsonl_path(split)) as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def to_training_row(r: dict) -> dict:
    # TRL wants a "prompt" column (list of chat messages); every other column goes to env.reset().
    return {
        "prompt": r["responses_create_params"]["input"],
        "ground_truth": r["ground_truth"],
        "category": r["category"],
        "row_id": r["id"],
    }


# --------------------------------------------------------------------------- self-test
def self_test(limit: int):
    from rl.workplace_assistant.utils import get_tools
    from rl.workplace_env import TOOLKITS, WorkplaceEnv

    # 1. every tool method matches its original schema (names, params, required-ness)
    schemas = {s["name"]: s for s in get_tools(TOOLKITS)["schemas"]}
    bad = []
    for name, s in schemas.items():
        m = getattr(WorkplaceEnv, name, None)
        if m is None:
            bad.append((name, "missing method"))
            continue
        params = inspect.signature(m).parameters
        req = {p for p, v in params.items() if p != "self" and v.default is inspect.Parameter.empty}
        if set(params) - {"self"} != set(s["parameters"]["properties"]) or req != set(s["parameters"].get("required", [])):
            bad.append((name, "signature differs from schema"))
    public = {n for n in dir(WorkplaceEnv) if not n.startswith("_") and callable(getattr(WorkplaceEnv, n))}
    extra = public - set(schemas) - {"reset"}
    assert not bad and not extra, f"schema mismatches: {bad}; unexpected public methods: {extra}"
    print(f"ok: {len(schemas)} tool methods match schemas; public surface = tools + reset")

    # 2. ground-truth replay through the tool methods scores 1.0; empty rollout scores 1.0 iff gt empty
    n = fail_replay = noop_hits = noop_expected = 0
    for split in ("validation", "train"):
        for r in load_rows(split, limit):
            n += 1
            kw = to_training_row(r)
            env = WorkplaceEnv()
            env.reset(**kw)
            for call in r["ground_truth"]:
                out = getattr(env, call["name"])(**json.loads(call["arguments"]))
                if isinstance(out, str) and out.startswith("Error executing tool"):
                    print("tool error on ground-truth call:", r["id"], call["name"], out[:120])
            fail_replay += env._score() != 1.0
            empty = WorkplaceEnv()
            empty.reset(**kw)
            noop_hits += empty._score() == 1.0
            noop_expected += len(r["ground_truth"]) == 0
    print(f"rows={n} ground-truth replay failures={fail_replay} "
          f"empty-rollout reward=1 on {noop_hits} rows (empty-ground-truth rows: {noop_expected})")
    assert fail_replay == 0, "ground-truth replay through the env did not score 1.0"
    assert noop_hits >= noop_expected, "an empty ground truth should always be matched by doing nothing"
    print("self-test passed")


# --------------------------------------------------------------------------- preflight
def preflight():
    import transformers
    import trl
    from trl import GRPOConfig, GRPOTrainer

    print("trl", trl.__version__, "| transformers", transformers.__version__)
    need_cfg = ["max_tool_calling_iterations", "chat_template_kwargs", "num_generations",  # VERIFY
                "max_completion_length", "use_vllm"]
    have_cfg = set(inspect.signature(GRPOConfig).parameters)
    need_trainer = ["environment_factory", "reward_funcs", "peft_config"]  # VERIFY
    have_trainer = set(inspect.signature(GRPOTrainer.__init__).parameters)
    missing = [f"GRPOConfig.{p}" for p in need_cfg if p not in have_cfg] + \
              [f"GRPOTrainer.{p}" for p in need_trainer if p not in have_trainer]
    if missing:
        sys.exit(f"PREFLIGHT FAILED: installed TRL lacks: {missing}. Fix the script / TRL version first.")
    print("ok: GRPOConfig / GRPOTrainer expose the arguments this script uses")

    # What schema would the model see? (transformers' converter; TRL builds schemas the same way)
    from transformers.utils import get_json_schema

    from rl.workplace_env import WorkplaceEnv

    env = WorkplaceEnv()
    tools = [n for n in dir(env) if not n.startswith("_") and n != "reset" and callable(getattr(env, n))]
    sample = get_json_schema(getattr(env, "email_send_email"))
    print("tools exposed:", len(tools), "| 'reset' exposed?", False)
    print("sample schema:", json.dumps(sample)[:600])
    print("CHECK BY EYE: sample schema marks required/optional params sensibly; get_reward not a tool.")


# --------------------------------------------------------------------------- train
def train(a):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from rl.workplace_env import WorkplaceEnv, reward_func

    rows = load_rows("train", a.n_train)
    if a.drop_noop:
        rows = [r for r in rows if r["ground_truth"]]
    if not a.keep_degenerate:
        # Rows with a real instruction whose ground-truth calls leave the state unchanged (~0.5% of
        # the data): doing nothing would be rewarded, so they only add noise.
        import contextlib
        import io

        from rl.workplace_assistant.utils import is_correct

        def degenerate(r):
            with contextlib.redirect_stdout(io.StringIO()):
                return bool(r["ground_truth"]) and is_correct([], r["ground_truth"], None)

        before = len(rows)
        rows = [r for r in rows if not degenerate(r)]
        print(f"dropped {before - len(rows)} degenerate rows (non-empty ground truth, no state change)")
    dataset = Dataset.from_list([to_training_row(r) for r in rows])
    floor = sum(1 for r in rows if not r["ground_truth"]) / len(rows)
    print(f"{len(rows)} training rows; a do-nothing policy scores {floor:.1%} (the floor to beat)")

    config = GRPOConfig(
        output_dir=a.out,
        max_steps=a.max_steps,
        num_generations=a.num_generations,                    # group size; must divide the generation batch
        per_device_train_batch_size=a.num_generations,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        max_completion_length=a.max_completion_length,        # VERIFY: total tokens across ALL turns
        max_tool_calling_iterations=6,                        # VERIFY: matches NeMo Gym simple_agent max_steps
        chat_template_kwargs={"enable_thinking": False},      # VERIFY: Qwen3 thinking off (history strips it anyway)
        use_vllm=a.use_vllm,
        log_completions=True,
        logging_steps=1,
        save_steps=a.save_steps,
        bf16=True,
        report_to="none",
    )
    peft_config = None
    if a.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")

    trainer = GRPOTrainer(
        model=a.model,
        args=config,
        train_dataset=dataset,                                # VERIFY: columns other than "prompt" reach env.reset()
        reward_funcs=reward_func,                             # reads env._score(); no public get_reward on purpose
        environment_factory=WorkplaceEnv,                     # VERIFY: pass the class; one instance per rollout
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(a.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true", help="download the dataset from the HF hub first")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--self-test-rows", type=int, default=150)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--out", default="out_workplace_smoke")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--drop-noop", action="store_true", help="drop empty-ground-truth rows")
    p.add_argument("--keep-degenerate", action="store_true", help="keep rows whose ground truth changes nothing")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-completion-length", type=int, default=3072)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--use-vllm", action="store_true", help="only after confirming vLLM supports the model")
    p.add_argument("--no-lora", dest="lora", action="store_false")
    a = p.parse_args()

    if a.fetch:
        fetch()
    ran = False
    if a.self_test:
        self_test(a.self_test_rows)
        ran = True
    if a.preflight:
        preflight()
        ran = True
    if not ran:
        train(a)


if __name__ == "__main__":
    main()
