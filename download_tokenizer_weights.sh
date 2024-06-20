export HF_ENDPOINT=https://hf-mirror.com

mkdir checkpoints
cd checkpoints

# download tokenizer weights
huggingface-cli download TencentARC/Open-MAGVIT2 --local-dir magvit2
