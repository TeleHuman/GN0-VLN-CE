from dataclasses import dataclass


@dataclass
class EvalRuntimeConfig:
    task: str
    model_path: str
    output_dir: str
    habitat_config_path: str
    eval_split: str = "val_unseen"
    dataset_data_path: str = ""
    scenes_dir: str = ""
    split_num: int = 1
    split_id: int = 0
    start_idx: int = 0
    end_idx: int = -1
    max_episodes: int = 0
    max_steps_per_episode: int = 500
    early_stop_rotation: int = 25
    resume: bool = False
    prompt_type: str = "V3HF"
    action_num: int = 1
    load_dtype: str = "bf16"
    max_new_tokens: int = 512
    history_len: int = 16
    device: str = "cuda:0"
    fallback_action: int = 0
