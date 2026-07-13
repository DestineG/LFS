import os

from src.utils.common import get_preprocess_pretrain_data_dir
from src.data.download.config import PRETRAIN_DATASETS

def generate_preprocess_dataset_dir(local_dir: str):
    if os.path.exists(local_dir) and not os.path.isdir(local_dir):
        raise ValueError(f"{local_dir} exists but is not a directory.")
    dataset_name = os.path.basename(os.path.normpath(local_dir))
    root_preprocess_dir = get_preprocess_pretrain_data_dir()
    preprocess_dataset_dir = os.path.join(root_preprocess_dir, dataset_name)
    os.makedirs(preprocess_dataset_dir, exist_ok=True)
    return preprocess_dataset_dir

preprocess_configs = []

for dataset_config in PRETRAIN_DATASETS:
    repo_id = dataset_config.get("repo_id", "")
    revision = dataset_config.get("revision", "main")
    local_dir = dataset_config.get("local_dir", "")
    rows_per_group = dataset_config.get("rows_per_group", 1024)
    max_row_groups_per_file = dataset_config.get("max_row_groups_per_file", 16)
    compression = dataset_config.get("compression", "zstd")
    text_column = dataset_config.get("text_column", "text")

    preprocess_config = {
        "repo_id": repo_id,
        "revision": revision,
        "local_dir": local_dir,
        "preprocess_dir": generate_preprocess_dataset_dir(local_dir),
        "rows_per_group": rows_per_group,
        "max_row_groups_per_file": max_row_groups_per_file,
        "compression": compression,
        "text_column": text_column,
    }
    preprocess_configs.append(preprocess_config)

for config in preprocess_configs:
    print(f"Preprocess config for {config['repo_id']}:")
    print(f"  Revision: {config['revision']}")
    print(f"  Local directory: {config['local_dir']}")
    print(f"  Preprocess directory: {config['preprocess_dir']}")
    print(f"  Rows per group: {config['rows_per_group']}")
    print(f"  Max row groups per file: {config['max_row_groups_per_file']}")
    print(f"  Compression: {config['compression']}")
    print(f"  Text column: {config['text_column']}")
    print()