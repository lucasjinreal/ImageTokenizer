export HF_ENDPOINT=https://hf-mirror.com

mkdir checkpoints
cd checkpoints

# download tokenizer weights
huggingface-cli download TencentARC/Open-MAGVIT2 --local-dir magvit2
huggingface-cli download fun-research/TiTok --local-dir titok

wget $HF_ENDPOINT/Daniel0724/OmniTokenizer/resolve/main/imagenet_sthv2.ckpt -o omni_imagenet_sthv2.ckpt