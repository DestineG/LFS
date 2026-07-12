from huggingface_hub import HfApi, RepoFile, RepoFolder


def list_hf_dataset_files(
    repo_id: str,
    path_in_repo: str,
    revision: str = "main",
) -> list[str]:
    api = HfApi()

    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo=path_in_repo,
        recursive=False,
    )

    return [
        entry.path
        for entry in entries
        if isinstance(entry, RepoFile)
    ]

def list_hf_dataset_directories(
    repo_id: str,
    revision: str = "main",
) -> list[str]:
    api = HfApi()

    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=False,
    )

    return [
        entry.path
        for entry in entries
        if isinstance(entry, RepoFolder)
    ]
