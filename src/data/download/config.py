from src.utils.common import get_base_dir

BASE_DIR = get_base_dir()

# 预训练数据集配置
PRETRAIN_DATASETS = [
    {
        "repo_id": "karpathy/climbmix-400b-shuffle",
        "revision": "main",
        "selected_sub_dir": [
            ("", 2500),
        ],
        "file_type": "parquet",
        "local_dir": f"{BASE_DIR}/data/raw/karpathy--climbmix-400b-shuffle",
        "corpus": "en",
    },
    {
        "repo_id": "opencsg/chinese-fineweb-edu",
        "revision": "main",
        "selected_sub_dir": [
            ("IndustryCorpus", -1),
            ("Skypile", -1),
            ("cci2", -1),
            ("map", -1),
            ("tele", -1),
        ],
        "file_type": "parquet",
        "local_dir": f"{BASE_DIR}/data/raw/opencsg--chinese-fineweb-edu",
        "corpus": "zh",
    },
    {
        "repo_id": "bigcode/starcoderdata",
        "revision": "main",
        "selected_sub_dir": [
            ("c", -1),
            ("cuda", -1),
            ("cpp", -1),
            ("go", -1),
            ("java", -1),
            ("javascript", -1),
            ("python", -1),
        ],
        "file_type": "parquet",
        "local_dir": f"{BASE_DIR}/data/raw/data/bigcode--starcoderdata",
        "corpus": "code",
    },
    {
        "repo_id": "open-web-math/open-web-math",
        "revision": "main",
        "selected_sub_dir": [
            ("data", -1),
        ],
        "file_type": "parquet",
        "local_dir": f"{BASE_DIR}/data/raw/open-web-math--open-web-math",
        "corpus": "math",
    }
]

# 后训练数据集配置
POSTTRAIN_DATASETS = [
    {
        "repo_id": "opencsg/Fineweb-Edu-Chinese-V2.2",
        "revision": "main",
        "selected_sub_dir": [
            ("sft/cleaned", -1),
        ],
        "file_type": "jsonl",
        "local_dir": f"{BASE_DIR}/data/raw/opencsg--Fineweb-Edu-Chinese-V2.2",
    },
    {
        "repo_id": "HuggingFaceTB/smol-smoltalk",
        "revision": "main",
        "selected_sub_dir": [
            ("data", -1),
        ],
        "file_type": "parquet",
        "local_dir": f"{BASE_DIR}/data/raw/HuggingFaceTB--smol-smoltalk",
    },
    {
        "repo_id": "cais/mmlu",
        "revision": "main",
        "selected_sub_dir": [
        ],
        "file_type": "parquet",
        "local_dir": f"{BASE_DIR}/data/raw/cais--mmlu",
    },
    {
        "repo_id": "openai/openai_humaneval",
        "revision": "main",
        "selected_sub_dir": [
            ("openai_humaneval", -1),
        ],
        "file_type": "parquet",
        "local_dir": f"{BASE_DIR}/data/raw/openai--openai_humaneval",
    },
    {
        "repo_id": "openai/gsm8k",
        "revision": "main",
        "selected_sub_dir": [
            ("main", -1),
            ("socratic", -1)
        ],
        "file_type": "parquet",
        "local_dir": f"{BASE_DIR}/data/raw/openai--gsm8k",
    },
    {
        "repo_id": "allenai/ai2_arc",
        "revision": "main",
        "selected_sub_dir": [
            ("ARC-Easy", -1),
            ("ARC-Challenge", -1),
        ],
        "file_type": "parquet",
        "local_dir": f"{BASE_DIR}/data/raw/allenai--ai2_arc",
    }
]