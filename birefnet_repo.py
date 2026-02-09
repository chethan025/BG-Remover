from huggingface_hub import snapshot_download

# This downloads the model weights AND the python code scripts
model_path = snapshot_download(
    repo_id="zhengpeng7/BiRefNet", 
    local_dir="./BiRefNet",
    local_dir_use_symlinks=False
)

print(f"Model downloaded to: {model_path}")