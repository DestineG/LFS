import json
from pathlib import Path

from .config import POSTTRAIN_DATASETS, PRETRAIN_DATASETS
from src.utils.common import get_base_dir, log_info, log_warning
from .utils import list_hf_dataset_files, list_hf_dataset_directories

def get_manifests_dir():
    manifests_dir = Path(get_base_dir()) / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    return manifests_dir

def get_manifest_path(repo_id: str, revision: str):
    manifest_path = get_manifests_dir() / f"{repo_id.replace('/', '--')}_{revision}.json"
    return manifest_path

def save_download_manifest(download_config, overwrite: bool = True):
    repo_id = download_config["repo_id"]
    revision = download_config["revision"]
    manifest_path = get_manifest_path(repo_id, revision)
    if manifest_path.exists():
        if not overwrite:
            log_warning(f"Manifest file {manifest_path} already exists and overwrite is set to False. Skipping save.")
            return str(manifest_path)
        else:
            log_warning(f"Manifest file {manifest_path} already exists and will be overwritten.")

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(download_config, f, ensure_ascii=False, indent=2)
    return str(manifest_path)

def load_download_manifest(repo_id: str, revision: str):
    manifest_path = get_manifest_path(repo_id, revision)
    if not manifest_path.exists():
        log_warning(f"Manifest file {manifest_path} does not exist.")
        return None

    with manifest_path.open("r", encoding="utf-8") as f:
        download_config = json.load(f)
    return download_config

def generate_download_config(dataset_config, persistent: bool = True, overwrite: bool = True):
    repo_id = dataset_config["repo_id"]
    revision = dataset_config.get("revision", "main")
    selected_sub_dir = dataset_config.get("selected_sub_dir", [])
    file_type = dataset_config.get("file_type", "parquet")
    local_dir = dataset_config.get("local_dir", None)
    if local_dir is None:
        raise ValueError(f"local_dir must be provided for dataset {repo_id}")
    
    loaded_cfg = load_download_manifest(repo_id, revision)
    if loaded_cfg is not None and not overwrite:
        log_warning(f"Manifest for {repo_id} at revision {revision} already exists and overwrite is set to False. Skipping generation.")
        return loaded_cfg

    if len(selected_sub_dir) == 0:
        directories = list_hf_dataset_directories(
            repo_id=repo_id,
            revision=revision,
        )
        selected_sub_dir = [("", -1)] + [(directory, -1) for directory in directories]
    
    res = {
        "repo_id": repo_id,
        "revision": revision,
        "local_dir": local_dir,
        "file_type": file_type,
        "max_num_files": 0,
        "download_num_files": 0,
        "path_map": [],
    }

    selected_files = set()

    for sub_dir, max_files in selected_sub_dir:
        files = list_hf_dataset_files(repo_id=repo_id, path_in_repo=sub_dir, revision=revision)

        typed_files = [
            file
            for file in files
            if file.lower().endswith(f".{file_type.lower()}")
        ]
        typed_files.sort()

        res["max_num_files"] += len(typed_files)

        clip_files = typed_files[:max_files] if max_files > 0 else typed_files

        for file in clip_files:
            if file in selected_files:
                continue

            selected_files.add(file)
            local_path = Path(local_dir) / file
            res["path_map"].append((file, str(local_path)))
    
    res["download_num_files"] = len(res["path_map"])

    if persistent:
        save_download_manifest(res, overwrite)

    return res

def generate_pretrain_download_configs(persistent: bool = True, overwrite: bool = True):
    configs = []
    for dataset_config in PRETRAIN_DATASETS:
        config = generate_download_config(dataset_config, persistent, overwrite)
        configs.append(config)
    return configs

def generate_posttrain_download_configs(persistent: bool = True, overwrite: bool = True):
    configs = []
    for dataset_config in POSTTRAIN_DATASETS:
        config = generate_download_config(dataset_config, persistent, overwrite)
        configs.append(config)
    return configs

def get_all_download_configs(persistent: bool = True, overwrite: bool = True):
    pretrain_configs = generate_pretrain_download_configs(persistent, overwrite)
    posttrain_configs = generate_posttrain_download_configs(persistent, overwrite)
    return pretrain_configs + posttrain_configs

if __name__ == "__main__":
    cfgs = get_all_download_configs(persistent=True, overwrite=True)
    log_info(f"Generated {len(cfgs)} download configs.")