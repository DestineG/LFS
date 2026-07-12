import time
from pathlib import Path
import requests
from multiprocessing import Pool

from src.utils.common import log_info, log_warning, log_error
from .generate_download_config import get_all_download_configs, load_download_manifest


def download_single_file(url: str, local_path: str) -> bool:
    """通过完整 URL 下载单个文件到指定本地路径。"""

    filepath = Path(local_path)
    temp_path = Path(f"{local_path}.tmp")

    # 自动创建本地目录
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 文件已存在则跳过
    if filepath.exists():
        print(f"Skipping {filepath} (already exists)")
        return True

    print(f"Downloading {url}")
    print(f"Saving to {filepath}")

    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()

                with temp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            # 下载完整后再替换为最终文件
            temp_path.replace(filepath)

            print(f"Successfully downloaded {filepath}")
            return True

        except (requests.RequestException, OSError) as error:
            print(
                f"Attempt {attempt}/{max_attempts} failed "
                f"for {url}: {error}"
            )

            # 清理未完成的临时文件
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(
                    f"Failed to download {url} "
                    f"after {max_attempts} attempts"
                )
                return False

    return False

def download_single_file_worker(args):
    url, local_path = args
    return download_single_file(url, local_path)

def download_single_dataset(image_url: str, repo_id: str, revision: str, num_workers: int = 1):
    if num_workers <= 0:
        raise ValueError(f"num_workers must be greater than 0, got {num_workers}")

    base_url = f"{image_url}/datasets/{repo_id}/resolve/{revision}"

    config = load_download_manifest(repo_id, revision)
    if config is None:
        log_error(f"No download manifest found for {repo_id} at revision {revision}")
        raise ValueError(f"No download manifest found for {repo_id} at revision {revision}")
    
    download_num_files = config.get("download_num_files", 0)
    if download_num_files <= 0:
        log_error(f"Invalid download_num_files ({download_num_files}) in manifest for {repo_id} at revision {revision}")
        raise ValueError(f"Invalid download_num_files ({download_num_files}) in manifest for {repo_id} at revision {revision}")

    path_map = config.get("path_map", [])
    if not path_map:
        log_error(f"No path_map found in manifest for {repo_id} at revision {revision}")
        raise ValueError(f"No path_map found in manifest for {repo_id} at revision {revision}")

    if download_num_files != len(path_map):
        log_error(f"Mismatch between download_num_files ({download_num_files}) and path_map length ({len(path_map)}) for {repo_id} at revision {revision}")
        raise ValueError(f"Mismatch between download_num_files ({download_num_files}) and path_map length ({len(path_map)}) for {repo_id} at revision {revision}")

    url_path_pairs = [
        (f"{base_url}/{remote_path}", local_path)
        for remote_path, local_path in path_map
    ]

    worker_count = min(num_workers, download_num_files)

    log_info(
        f"Starting download dataset {repo_id}: "
        f"total={download_num_files}, workers={worker_count}"
    )

    success_num = 0
    failed_num = 0
    completed_num = 0

    # 大约每完成 5% 打印一次进度
    log_interval = max(1, download_num_files // 20)

    with Pool(processes=worker_count) as pool:
        results = pool.imap_unordered(download_single_file_worker, url_path_pairs)

        for success in results:
            completed_num += 1

            if success:
                success_num += 1
            else:
                failed_num += 1

            if completed_num % log_interval == 0 or completed_num == download_num_files:
                progress = completed_num / download_num_files * 100

                log_info(
                    f"Downloading dataset {repo_id}: "
                    f"progress={completed_num}/{download_num_files} "
                    f"({progress:.1f}%), "
                    f"success={success_num}, failed={failed_num}"
                )
    return success_num

def download_datasets(image_url: str, configs: list[dict], num_workers: int = 1):
    if num_workers <= 0:
        raise ValueError(f"num_workers must be greater than 0, got {num_workers}")

    results = {}

    for config in configs:
        repo_id = config["repo_id"]
        revision = config.get("revision", "main")
        download_num_files = config.get("download_num_files", 0)

        try:
            success_num = download_single_dataset(image_url, repo_id, revision, num_workers)
            failed_num = download_num_files - success_num

            results[repo_id] = {
                "revision": revision,
                "total_num": download_num_files,
                "success_num": success_num,
                "failed_num": failed_num,
            }

        except Exception as error:
            log_error(f"Failed to download dataset {repo_id}: {error}")

            results[repo_id] = {
                "revision": revision,
                "total_num": download_num_files,
                "success_num": 0,
                "failed_num": download_num_files,
            }

    return results

if __name__ == "__main__":
    # image_url = "https://huggingface.co"
    image_url = "https://hf-mirror.com"
    configs = get_all_download_configs(persistent=True, overwrite=False)

    results = download_datasets(image_url, configs, num_workers=8)

    for repo_id, result in results.items():
        log_info(
            f"{repo_id}: "
            f"total={result['total_num']}, "
            f"success={result['success_num']}, "
            f"failed={result['failed_num']}"
        )