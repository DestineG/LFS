from src.utils.common import get_raw_data_dir

RAW_DATA_DIR = get_raw_data_dir()

# 预训练数据集配置
PRETRAIN_DATASETS = [
    {
        "repo_id": "karpathy/climbmix-400b-shuffle",
        "revision": "main",
        "selected_sub_dir": [
            ("", 2500),
        ],
        "file_type": "parquet",
        "local_dir": f"{RAW_DATA_DIR}/karpathy--climbmix-400b-shuffle",
        "corpus": "en",
        "text_column": "text",
        "rows_per_group": 8192,
        "max_row_groups_per_file": 32,
        "compression": "zstd",
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
        "local_dir": f"{RAW_DATA_DIR}/opencsg--chinese-fineweb-edu",
        "corpus": "zh",
        "text_column": "text",
        "rows_per_group": 4096,
        "max_row_groups_per_file": 32,
        "compression": "zstd",
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
        "local_dir": f"{RAW_DATA_DIR}/bigcode--starcoderdata",
        "corpus": "code",
        "text_column": "content",
        "rows_per_group": 2048,
        "max_row_groups_per_file": 32,
        "compression": "zstd",
    },
    {
        "repo_id": "open-web-math/open-web-math",
        "revision": "main",
        "selected_sub_dir": [
            ("data", -1),
        ],
        "file_type": "parquet",
        "local_dir": f"{RAW_DATA_DIR}/open-web-math--open-web-math",
        "corpus": "math",
        "text_column": "text",
        "rows_per_group": 4096,
        "max_row_groups_per_file": 32,
        "compression": "zstd",
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
        "local_dir": f"{RAW_DATA_DIR}/opencsg--Fineweb-Edu-Chinese-V2.2",
    },
    {
        "repo_id": "HuggingFaceTB/smol-smoltalk",
        "revision": "main",
        "selected_sub_dir": [
            ("data", -1),
        ],
        "file_type": "parquet",
        "local_dir": f"{RAW_DATA_DIR}/HuggingFaceTB--smol-smoltalk",
    },
    {
        "repo_id": "cais/mmlu",
        "revision": "main",
        "selected_sub_dir": [
        ],
        "file_type": "parquet",
        "local_dir": f"{RAW_DATA_DIR}/cais--mmlu",
    },
    {
        "repo_id": "openai/openai_humaneval",
        "revision": "main",
        "selected_sub_dir": [
            ("openai_humaneval", -1),
        ],
        "file_type": "parquet",
        "local_dir": f"{RAW_DATA_DIR}/openai--openai_humaneval",
    },
    {
        "repo_id": "openai/gsm8k",
        "revision": "main",
        "selected_sub_dir": [
            ("main", -1),
            ("socratic", -1)
        ],
        "file_type": "parquet",
        "local_dir": f"{RAW_DATA_DIR}/openai--gsm8k",
    },
    {
        "repo_id": "allenai/ai2_arc",
        "revision": "main",
        "selected_sub_dir": [
            ("ARC-Easy", -1),
            ("ARC-Challenge", -1),
        ],
        "file_type": "parquet",
        "local_dir": f"{RAW_DATA_DIR}/allenai--ai2_arc",
    }
]