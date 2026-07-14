import json
import os
from datetime import datetime, timezone
from typing import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.download.config import PRETRAIN_DATASETS
from src.utils.common import (
    get_preprocess_pretrain_data_dir,
    log_error,
    log_info,
    log_warning,
)
from .utils import find_files


META_FILENAME = "meta.jsonl"
OUTPUT_FILE_PREFIX = "part-"
DEFAULT_PROGRESS_EVERY_FILES = 10


def generate_preprocess_dataset_dir(local_dir: str) -> str:
    """
    Generate the output directory for a preprocessed dataset.

    Example:
        local_dir:
            data/raw/karpathy--climbmix-400b-shuffle

        preprocess_dataset_dir:
            data/preprocessed/karpathy--climbmix-400b-shuffle
    """
    if not isinstance(local_dir, str) or not local_dir.strip():
        raise ValueError("local_dir must be a non-empty string.")

    if os.path.exists(local_dir) and not os.path.isdir(local_dir):
        raise ValueError(
            f"{local_dir} exists but is not a directory."
        )

    dataset_name = os.path.basename(
        os.path.normpath(local_dir)
    )

    if not dataset_name:
        raise ValueError(
            f"Unable to determine dataset name from local_dir: "
            f"{local_dir}"
        )

    root_preprocess_dir = get_preprocess_pretrain_data_dir()

    preprocess_dataset_dir = os.path.join(
        root_preprocess_dir,
        dataset_name,
    )

    os.makedirs(
        preprocess_dataset_dir,
        exist_ok=True,
    )

    return preprocess_dataset_dir


def generate_preprocess_configs() -> list[dict]:
    """
    Generate preprocessing configurations for all pretraining datasets.
    """
    preprocess_configs = []

    for dataset_config in PRETRAIN_DATASETS:
        repo_id = dataset_config.get(
            "repo_id",
            "",
        )
        revision = dataset_config.get(
            "revision",
            "main",
        )
        file_type = dataset_config.get(
            "file_type",
            "parquet",
        )
        local_dir = dataset_config.get(
            "local_dir",
            "",
        )
        rows_per_group = dataset_config.get(
            "rows_per_group",
            1024,
        )
        max_row_groups_per_file = dataset_config.get(
            "max_row_groups_per_file",
            16,
        )
        compression = dataset_config.get(
            "compression",
            "zstd",
        )
        text_column = dataset_config.get(
            "text_column",
            "text",
        )
        corpus = dataset_config.get(
            "corpus",
            "",
        )
        progress_every_files = dataset_config.get(
            "progress_every_files",
            DEFAULT_PROGRESS_EVERY_FILES,
        )

        file_paths = sorted(
            find_files(
                local_dir,
                f"*.{file_type}",
            )
        )

        preprocess_config = {
            "repo_id": repo_id,
            "revision": revision,
            "file_type": file_type,
            "local_dir": local_dir,
            "preprocess_dir": generate_preprocess_dataset_dir(
                local_dir
            ),
            "rows_per_group": rows_per_group,
            "max_row_groups_per_file": (
                max_row_groups_per_file
            ),
            "compression": compression,
            "text_column": text_column,
            "corpus": corpus,
            "progress_every_files": progress_every_files,
            "file_paths": file_paths,
        }

        preprocess_configs.append(
            preprocess_config
        )

    return preprocess_configs


def _iter_text_tables(
    file_paths: list[str],
    text_column: str,
    batch_size: int,
    dataset_name: str,
    progress_every_files: int,
) -> Iterator[pa.Table]:
    """
    Stream text batches from multiple Parquet files.

    Each yielded Table is a reading batch rather than a physical
    Parquet Row Group.

    Progress is logged after every fixed number of completed source files:

        dataset_name: completed_files/total_files
    """
    if (
        not isinstance(progress_every_files, int)
        or progress_every_files <= 0
    ):
        raise ValueError(
            f"Invalid progress_every_files: "
            f"{progress_every_files}. "
            f"It must be a positive integer."
        )

    total_files = len(file_paths)
    completed_files = 0

    for file_path in file_paths:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Input file does not exist: {file_path}"
            )

        try:
            parquet_file = pq.ParquetFile(
                file_path
            )
        except Exception as exc:
            log_error(
                f"Failed to open source file: "
                f"dataset={dataset_name}, "
                f"file={file_path}, "
                f"error={exc}"
            )
            raise RuntimeError(
                f"Failed to open Parquet file: "
                f"{file_path}"
            ) from exc

        available_columns = (
            parquet_file.schema_arrow.names
        )

        if text_column not in available_columns:
            raise ValueError(
                f"Text column {text_column!r} was not found "
                f"in {file_path}. "
                f"Available columns: {available_columns}"
            )

        try:
            batches = parquet_file.iter_batches(
                batch_size=batch_size,
                columns=[text_column],
                use_threads=True,
            )

            for batch in batches:
                if batch.num_rows == 0:
                    continue

                table = pa.Table.from_batches(
                    [batch]
                )

                column = table.column(
                    text_column
                )

                # Normalize different source schemas, such as
                # string and large_string, to a common string type.
                if column.type != pa.string():
                    try:
                        column = column.cast(
                            pa.string()
                        )
                    except Exception as exc:
                        raise TypeError(
                            f"Column {text_column!r} in "
                            f"{file_path} cannot be converted "
                            f"from {column.type} to string."
                        ) from exc

                yield pa.table(
                    {
                        text_column: column,
                    },
                    schema=pa.schema([
                        pa.field(
                            text_column,
                            pa.string(),
                        ),
                    ]),
                )

        except Exception as exc:
            log_error(
                f"Failed to read source file: "
                f"dataset={dataset_name}, "
                f"file={file_path}, "
                f"error={exc}"
            )
            raise RuntimeError(
                f"Failed while reading Parquet file: "
                f"{file_path}"
            ) from exc

        completed_files += 1

        should_log_progress = (
            completed_files % progress_every_files == 0
            or completed_files == total_files
        )

        if should_log_progress:
            log_info(
                f"{dataset_name}: "
                f"{completed_files}/{total_files}"
            )


def _remove_previous_outputs(
    preprocess_dir: str,
) -> None:
    """
    Remove Parquet shards and metadata generated by a previous run.

    Unrelated files in the preprocessing directory are preserved.
    """
    removed_files = 0

    for filename in os.listdir(
        preprocess_dir
    ):
        is_output_shard = (
            filename.startswith(
                OUTPUT_FILE_PREFIX
            )
            and filename.endswith(
                ".parquet"
            )
        )

        is_metadata = filename in {
            META_FILENAME,
            f"{META_FILENAME}.tmp",
        }

        if (
            not is_output_shard
            and not is_metadata
        ):
            continue

        file_path = os.path.join(
            preprocess_dir,
            filename,
        )

        try:
            os.remove(file_path)
            removed_files += 1
        except OSError as exc:
            raise RuntimeError(
                f"Failed to remove previous output: "
                f"{file_path}"
            ) from exc

    if removed_files > 0:
        log_warning(
            f"Removed {removed_files} existing "
            f"preprocessing output files from "
            f"{preprocess_dir}."
        )


def _write_metadata(
    preprocess_dir: str,
    dataset_metadata: dict,
    shard_metadata: list[dict],
) -> str:
    """
    Write dataset-level and shard-level metadata to meta.jsonl.

    The first line records the complete dataset summary.
    Each following line records one output Parquet shard.
    """
    metadata_path = os.path.join(
        preprocess_dir,
        META_FILENAME,
    )

    temporary_path = (
        f"{metadata_path}.tmp"
    )

    try:
        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    dataset_metadata,
                    ensure_ascii=False,
                )
                + "\n"
            )

            for shard in shard_metadata:
                file.write(
                    json.dumps(
                        shard,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # Replace the metadata file only after the temporary file
        # has been written successfully.
        os.replace(
            temporary_path,
            metadata_path,
        )

    except Exception:
        if os.path.exists(
            temporary_path
        ):
            try:
                os.remove(
                    temporary_path
                )
            except OSError:
                pass

        raise

    return metadata_path


def preprocess_dataset(
    preprocess_config: dict,
) -> dict:
    """
    Merge source Parquet files into one logical preprocessed dataset.

    Output layout:
        preprocess_dir/
        ├── part-00000.parquet
        ├── part-00001.parquet
        ├── ...
        └── meta.jsonl

    Each output file contains at most
    ``max_row_groups_per_file`` Row Groups.

    Each Row Group contains ``rows_per_group`` rows, except for
    the final Row Group of the entire dataset.
    """
    repo_id = preprocess_config.get(
        "repo_id",
        "",
    )
    revision = preprocess_config.get(
        "revision",
        "main",
    )
    corpus = preprocess_config.get(
        "corpus",
        "",
    )
    local_dir = preprocess_config.get(
        "local_dir",
        "",
    )
    preprocess_dir = preprocess_config.get(
        "preprocess_dir",
        "",
    )
    file_paths = preprocess_config.get(
        "file_paths",
        [],
    )
    file_type = preprocess_config.get(
        "file_type",
        "parquet",
    )
    rows_per_group = preprocess_config.get(
        "rows_per_group"
    )
    max_row_groups_per_file = (
        preprocess_config.get(
            "max_row_groups_per_file"
        )
    )
    compression = preprocess_config.get(
        "compression",
        "zstd",
    )
    text_column = preprocess_config.get(
        "text_column"
    )
    progress_every_files = preprocess_config.get(
        "progress_every_files",
        DEFAULT_PROGRESS_EVERY_FILES,
    )

    if not repo_id:
        raise ValueError(
            "repo_id must not be empty."
        )

    if not preprocess_dir:
        raise ValueError(
            "preprocess_dir must not be empty."
        )

    if not os.path.isdir(
        preprocess_dir
    ):
        raise ValueError(
            f"Preprocess directory does not exist: "
            f"{preprocess_dir}"
        )

    if not file_paths:
        raise ValueError(
            f"No source files were found in "
            f"local directory: {local_dir}"
        )

    if file_type.lower() != "parquet":
        raise ValueError(
            f"Unsupported file type: {file_type}. "
            f"This function currently supports "
            f"only Parquet."
        )

    if (
        not isinstance(rows_per_group, int)
        or rows_per_group <= 0
    ):
        raise ValueError(
            f"Invalid rows_per_group: "
            f"{rows_per_group}. "
            f"It must be a positive integer."
        )

    if (
        not isinstance(
            max_row_groups_per_file,
            int,
        )
        or max_row_groups_per_file <= 0
    ):
        raise ValueError(
            f"Invalid max_row_groups_per_file: "
            f"{max_row_groups_per_file}. "
            f"It must be a positive integer."
        )

    if (
        not isinstance(text_column, str)
        or not text_column.strip()
    ):
        raise ValueError(
            f"Invalid text_column: {text_column}. "
            f"It must be a non-empty string."
        )

    if (
        not isinstance(compression, str)
        or not compression.strip()
    ):
        raise ValueError(
            f"Invalid compression: {compression}. "
            f"It must be a non-empty string."
        )

    if (
        not isinstance(progress_every_files, int)
        or progress_every_files <= 0
    ):
        raise ValueError(
            f"Invalid progress_every_files: "
            f"{progress_every_files}. "
            f"It must be a positive integer."
        )

    file_paths = sorted(
        file_paths
    )

    output_schema = pa.schema([
        pa.field(
            text_column,
            pa.string(),
        ),
    ])

    _remove_previous_outputs(
        preprocess_dir
    )

    writer: pq.ParquetWriter | None = None
    current_output_path: str | None = None

    output_file_index = 0
    row_groups_in_current_file = 0
    rows_in_current_file = 0

    total_input_files = len(
        file_paths
    )
    total_rows = 0
    total_row_groups = 0
    total_output_files = 0

    pending_tables: list[pa.Table] = []
    pending_rows = 0

    shard_metadata: list[dict] = []

    log_info(
        f"Starting preprocessing: "
        f"repo_id={repo_id}, "
        f"input_files={total_input_files}, "
        f"rows_per_group={rows_per_group}, "
        f"max_row_groups_per_file="
        f"{max_row_groups_per_file}, "
        f"progress_every_files="
        f"{progress_every_files}, "
        f"output_dir={preprocess_dir}"
    )

    def open_writer() -> pq.ParquetWriter:
        nonlocal current_output_path
        nonlocal total_output_files

        output_filename = (
            f"{OUTPUT_FILE_PREFIX}"
            f"{output_file_index:05d}.parquet"
        )

        current_output_path = os.path.join(
            preprocess_dir,
            output_filename,
        )

        total_output_files += 1

        log_info(
            f"Creating output shard: "
            f"{current_output_path}"
        )

        return pq.ParquetWriter(
            where=current_output_path,
            schema=output_schema,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )

    def close_writer() -> None:
        nonlocal writer
        nonlocal current_output_path
        nonlocal output_file_index
        nonlocal row_groups_in_current_file
        nonlocal rows_in_current_file

        if (
            writer is None
            or current_output_path is None
        ):
            return

        writer.close()

        file_size_bytes = os.path.getsize(
            current_output_path
        )

        output_filename = os.path.basename(
            current_output_path
        )

        shard_metadata.append({
            "record_type": "shard",
            "file_index": output_file_index,
            "file_name": output_filename,
            "relative_path": output_filename,
            "rows": rows_in_current_file,
            "row_groups": (
                row_groups_in_current_file
            ),
            "size_bytes": file_size_bytes,
            "compression": compression,
        })

        log_info(
            f"Completed output shard: "
            f"file={output_filename}, "
            f"rows={rows_in_current_file}, "
            f"row_groups="
            f"{row_groups_in_current_file}, "
            f"size_bytes={file_size_bytes}"
        )

        writer = None
        current_output_path = None

        output_file_index += 1
        row_groups_in_current_file = 0
        rows_in_current_file = 0

    def write_row_group(
        table: pa.Table,
    ) -> None:
        nonlocal writer
        nonlocal row_groups_in_current_file
        nonlocal rows_in_current_file
        nonlocal total_rows
        nonlocal total_row_groups

        if table.num_rows == 0:
            return

        if table.num_rows > rows_per_group:
            raise ValueError(
                f"A single output Row Group cannot "
                f"contain more than "
                f"{rows_per_group} rows, "
                f"but received {table.num_rows}."
            )

        if writer is None:
            writer = open_writer()

        writer.write_table(
            table,
            row_group_size=rows_per_group,
        )

        row_groups_in_current_file += 1
        rows_in_current_file += table.num_rows

        total_row_groups += 1
        total_rows += table.num_rows

        if (
            row_groups_in_current_file
            >= max_row_groups_per_file
        ):
            close_writer()

    try:
        for table in _iter_text_tables(
            file_paths=file_paths,
            text_column=text_column,
            batch_size=rows_per_group,
            dataset_name=repo_id,
            progress_every_files=(
                progress_every_files
            ),
        ):
            if table.num_rows == 0:
                continue

            pending_tables.append(
                table
            )
            pending_rows += table.num_rows

            # Source-file boundaries do not force new Row Groups.
            # Remaining rows can be combined with rows from the next
            # source file.
            while pending_rows >= rows_per_group:
                combined_table = pa.concat_tables(
                    pending_tables,
                    promote_options="default",
                )

                complete_row_group = (
                    combined_table.slice(
                        offset=0,
                        length=rows_per_group,
                    )
                )

                write_row_group(
                    complete_row_group
                )

                remaining_rows = (
                    combined_table.num_rows
                    - rows_per_group
                )

                if remaining_rows > 0:
                    remaining_table = (
                        combined_table.slice(
                            offset=rows_per_group,
                            length=remaining_rows,
                        )
                    )

                    pending_tables = [
                        remaining_table
                    ]
                    pending_rows = (
                        remaining_rows
                    )
                else:
                    pending_tables = []
                    pending_rows = 0

        # Write the final incomplete Row Group.
        if pending_rows > 0:
            final_table = pa.concat_tables(
                pending_tables,
                promote_options="default",
            )

            log_warning(
                f"Writing the final incomplete "
                f"Row Group with "
                f"{final_table.num_rows} rows. "
                f"Configured rows_per_group="
                f"{rows_per_group}."
            )

            write_row_group(
                final_table
            )

        close_writer()

        completed_at = datetime.now(
            timezone.utc
        ).isoformat()

        dataset_metadata = {
            "record_type": "dataset",
            "format_version": 1,
            "repo_id": repo_id,
            "revision": revision,
            "corpus": corpus,
            "source_dir": os.path.abspath(
                local_dir
            ),
            "preprocess_dir": os.path.abspath(
                preprocess_dir
            ),
            "file_type": file_type,
            "text_column": text_column,
            "compression": compression,
            "rows_per_group": rows_per_group,
            "max_row_groups_per_file": (
                max_row_groups_per_file
            ),
            "max_rows_per_file": (
                rows_per_group
                * max_row_groups_per_file
            ),
            "progress_every_files": (
                progress_every_files
            ),
            "input_files": total_input_files,
            "output_files": total_output_files,
            "total_rows": total_rows,
            "total_row_groups": (
                total_row_groups
            ),
            "schema": {
                field.name: str(field.type)
                for field in output_schema
            },
            "completed_at": completed_at,
        }

        metadata_path = _write_metadata(
            preprocess_dir=preprocess_dir,
            dataset_metadata=dataset_metadata,
            shard_metadata=shard_metadata,
        )

        log_info(
            f"Metadata written to: "
            f"{metadata_path}"
        )

        result = {
            **dataset_metadata,
            "metadata_path": metadata_path,
        }

        log_info(
            f"Preprocessing completed: "
            f"repo_id={repo_id}, "
            f"input_files={total_input_files}, "
            f"output_files={total_output_files}, "
            f"rows={total_rows}, "
            f"row_groups={total_row_groups}, "
            f"output_dir={preprocess_dir}"
        )

        return result

    except Exception as exc:
        if writer is not None:
            try:
                writer.close()
            except Exception as close_exc:
                log_error(
                    f"Failed to close the active "
                    f"Parquet writer: {close_exc}"
                )

        log_error(
            f"Preprocessing failed: "
            f"repo_id={repo_id}, "
            f"error={exc}"
        )

        raise


def preprocess_all_datasets(
    num_workers: int = 1,
) -> list[dict]:
    """
    Preprocess all datasets configured in PRETRAIN_DATASETS.

    Multiple datasets can be processed concurrently when
    ``num_workers`` is greater than 1.

    Args:
        num_workers:
            Maximum number of datasets processed concurrently.
            Set to 1 for sequential preprocessing.

    Returns:
        Successful preprocessing results.
    """
    if not isinstance(num_workers, int) or num_workers <= 0:
        raise ValueError(
            f"Invalid num_workers: {num_workers}. "
            f"It must be a positive integer."
        )

    preprocess_configs = generate_preprocess_configs()
    total_datasets = len(preprocess_configs)

    if total_datasets == 0:
        log_warning(
            "No dataset configurations were found for preprocessing."
        )
        return []

    # Avoid creating more worker processes than datasets.
    actual_num_workers = min(
        num_workers,
        total_datasets,
    )

    results: list[dict] = []
    failed_datasets: list[dict] = []

    log_info(
        f"Found {total_datasets} datasets to preprocess. "
        f"num_workers={actual_num_workers}"
    )

    # Keep the original sequential behavior when only one worker is used.
    if actual_num_workers == 1:
        for dataset_index, preprocess_config in enumerate(
            preprocess_configs,
            start=1,
        ):
            repo_id = preprocess_config.get(
                "repo_id",
                "",
            )

            log_info(
                f"Processing dataset "
                f"[{dataset_index}/{total_datasets}]: "
                f"{repo_id}"
            )

            try:
                result = preprocess_dataset(
                    preprocess_config
                )

                results.append(result)

                log_info(
                    f"Dataset completed "
                    f"[{dataset_index}/{total_datasets}]: "
                    f"{repo_id}"
                )

            except Exception as exc:
                failed_datasets.append({
                    "repo_id": repo_id,
                    "error": str(exc),
                })

                log_error(
                    f"Dataset preprocessing failed: "
                    f"repo_id={repo_id}, "
                    f"error={exc}"
                )

    else:
        # Process different datasets concurrently.
        with ProcessPoolExecutor(
            max_workers=actual_num_workers
        ) as executor:
            future_to_config = {
                executor.submit(
                    preprocess_dataset,
                    preprocess_config,
                ): preprocess_config
                for preprocess_config in preprocess_configs
            }

            completed_datasets = 0

            for future in as_completed(
                future_to_config
            ):
                preprocess_config = future_to_config[
                    future
                ]

                repo_id = preprocess_config.get(
                    "repo_id",
                    "",
                )

                completed_datasets += 1

                try:
                    result = future.result()
                    results.append(result)

                    log_info(
                        f"Dataset completed "
                        f"[{completed_datasets}/{total_datasets}]: "
                        f"{repo_id}"
                    )

                except Exception as exc:
                    failed_datasets.append({
                        "repo_id": repo_id,
                        "error": str(exc),
                    })

                    log_error(
                        f"Dataset preprocessing failed "
                        f"[{completed_datasets}/{total_datasets}]: "
                        f"repo_id={repo_id}, "
                        f"error={exc}"
                    )

    if failed_datasets:
        log_warning(
            f"Preprocessing finished with failures: "
            f"succeeded={len(results)}, "
            f"failed={len(failed_datasets)}, "
            f"total={total_datasets}."
        )

        for failure in failed_datasets:
            log_warning(
                f"Failed dataset: "
                f"repo_id={failure['repo_id']}, "
                f"error={failure['error']}"
            )
    else:
        log_info(
            f"All datasets were preprocessed successfully: "
            f"total={len(results)}."
        )

    return results


# uv run python -m src.data.preprocess.pretrain
if __name__ == "__main__":
    preprocess_all_datasets(
        num_workers=8,
    )
