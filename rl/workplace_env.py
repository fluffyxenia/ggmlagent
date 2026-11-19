"""TRL environment for the workplace_assistant task (for GRPOTrainer(environment_factory=...)).

Design notes:
  * Every PUBLIC method except `reset` becomes a model-visible tool in TRL. Only the 27 generated
    tool methods (rl/workplace_env_tools.py) and `reset` are public; everything else starts with
    "_" so it can't be exposed as a tool. In particular the reward is `_score()`, read through a
    reward function (`reward_func` below), not a public `get_reward`.
  * Tools run LIVE against this episode's in-memory pandas tables. The reward compares the final
    tables to the tables produced by applying the dataset's ground_truth calls to a fresh copy,
    using the same comparison as NeMo Gym's `is_correct` (case-insensitive except status /
    list_name / board). Reward is binary, like the original.
  * Rows with an empty ground_truth (conditional tasks) reward doing nothing: the ground-truth
    tables equal the initial tables.
"""
import json
import warnings

from rl.workplace_assistant.utils import execute_actions_and_reset_state, get_tools
from rl.workplace_env_tools import WorkplaceTools

TOOLKITS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]
_NO_LOWER = {"status", "list_name", "board"}
_TABLES = [
    ("calendar", "_calendar_events"),
    ("email", "_emails"),
    ("analytics", "_plots_data"),
    ("project_management", "_project_tasks"),
    ("customer_relationship_manager", "_crm_data"),
]


def _norm(df):
    df = df.copy()
    for col in df.columns:
        if col not in _NO_LOWER:
            df[col] = df[col].str.lower()
    return df


def states_match(env_a: dict, env_b: dict) -> bool:
    return all(
        _norm(getattr(env_a["containers"][c], attr)).equals(_norm(getattr(env_b["containers"][c], attr)))
        for c, attr in _TABLES
    )


class WorkplaceEnv(WorkplaceTools):
    def __init__(self):
        self._tool_env = get_tools(TOOLKITS)
        self._gt_env = None

    def reset(self, **kwargs):
        """Start an episode. Receives all dataset columns; uses `ground_truth`."""
        self._tool_env = get_tools(TOOLKITS)
        gt = kwargs.get("ground_truth")
        if isinstance(gt, str):
            gt = json.loads(gt)
        self._gt_env = execute_actions_and_reset_state(list(gt or []))
        return None

    def _call(self, name: str, **kwargs) -> str:
        args = {k: v for k, v in kwargs.items() if v is not None}
        try:
            result = self._tool_env["functions"][name](**args)
        except Exception as e:  # return the error to the model so it can correct itself
            return f"Error executing tool '{name}': {e}"
        return result if isinstance(result, str) else json.dumps(result, default=str)

    def _score(self) -> float:
        if self._gt_env is None:
            return 0.0
        try:
            return 1.0 if states_match(self._tool_env, self._gt_env) else 0.0
        except Exception as e:  # never crash a training run on a reward error, but be loud
            warnings.warn(f"workplace_env scoring error: {e!r}")
            return 0.0


def reward_func(environments, **kwargs) -> list:
    return [env._score() for env in environments]
