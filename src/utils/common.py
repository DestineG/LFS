import logging
import os
from pathlib import Path


def get_base_dir():
    data_dir = os.getenv("DATA")
    if data_dir is None:
        raise EnvironmentError("Environment variable DATA is not set")

    base_dir = os.path.join(data_dir, "LFS")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _build_logger():
    logger = logging.getLogger("LFS")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path(get_base_dir()) / "logs" / "lfs.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = _build_logger()


def log_info(message: str):
    logger.info(message, stacklevel=2)


def log_warning(message: str):
    logger.warning(message, stacklevel=2)

def log_error(message: str):
    logger.error(message, stacklevel=2)

if __name__ == "__main__":
    log_info("LFS logger initialized")
    log_warning("This is a warning")
    log_error("This is an error")