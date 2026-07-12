import os
def get_hf_dir():
    hf_dir = os.getenv("HF_HOME")
    if not hf_dir:
        hf_dir = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(hf_dir, exist_ok=True)
    return hf_dir

if __name__ == "__main__":
    print(get_hf_dir())