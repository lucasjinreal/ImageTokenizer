"""
Sending an image, encode it in a [1, 16, h, w] token
then decode it back to original image
"""

"""
We provide Tokenizer Inference code here.
"""
import os
import sys
import torch
import importlib
import numpy as np
from PIL import Image
import argparse
import torchvision.transforms as T
from imagetokenizer.model import Magvit2Tokenizer, OmniTokenizer, TiTok


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_vqgan_new(num_down, ckpt_path=None, is_gumbel=False):
    if "magvit2" in ckpt_path.lower():
        model = Magvit2Tokenizer(num_down=num_down, use_ema=True)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
    elif "omni" in ckpt_path.lower():
        model = OmniTokenizer()
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
    elif "titok" in ckpt_path.lower():
        model = TiTok()
        if ckpt_path is not None:
            model.load_weights(ckpt_path)
    return model.eval()


def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def get_image_tensor_for_encoder(image):
    image = image / 127.5 - 1.0
    image = T.ToTensor()(image).unsqueeze(0)
    # reshape the image to closest multiple 8 size
    height, width = image.shape[2], image.shape[3]
    new_height = ((height + 7) // 8) * 8
    new_width = ((width + 7) // 8) * 8  # 调整图像大小
    image = torch.nn.functional.interpolate(
        image, size=(new_height, new_width), mode="bilinear", align_corners=False
    )
    return image


def main(args):
    model = load_vqgan_new(args.num_down, args.ckpt_path).to(DEVICE)

    visualize_dir = "results/"
    visualize_version = "v0"
    visualize_original = os.path.join(
        visualize_dir, visualize_version, "original_{}".format(args.num_down)
    )
    visualize_rec = os.path.join(
        visualize_dir, visualize_version, "rec_{}".format(args.num_down)
    )
    if not os.path.exists(visualize_original):
        os.makedirs(visualize_original, exist_ok=True)

    if not os.path.exists(visualize_rec):
        os.makedirs(visualize_rec, exist_ok=True)

    img_f = args.image_file
    idx = os.path.basename(img_f)[:-4] + "_constructed"
    image_raw = Image.open(img_f)
    image = np.array(image_raw)
    print(f"original image size: {image.shape}")
    images_tensor = get_image_tensor_for_encoder(image)
    images_tensor = images_tensor.float().to(DEVICE)
    print(f"images: {images_tensor.shape}")

    quant, embedding, codebook_indices = model.encode(images_tensor)
    print(f"quant: {quant.shape}")
    print(f"embedding: {embedding.shape}")
    print(f"codebook_indices: {codebook_indices.shape}")
    reconstructed_images = model.decode(quant)

    image = images_tensor[0]
    reconstructed_image = reconstructed_images[0]

    image = custom_to_pil(image)
    reconstructed_image = custom_to_pil(reconstructed_image)
    reconstructed_image.resize((image_raw.width, image_raw.height))

    image.save(os.path.join(visualize_original, "{}.png".format(idx)))
    reconstructed_image.save(os.path.join(visualize_rec, "{}.png".format(idx)))


def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--num_down", default=3, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--image_file", default="images/a.jpg", type=str)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--tokenizer", default="magvit2")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
