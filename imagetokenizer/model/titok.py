import torch
import torch.nn as nn
from einops import rearrange
from typing import Mapping, Text, Tuple
import os

import torch
from einops import rearrange
from torch.cuda.amp import autocast

from .modules.titok_transformer import TiTokEncoder, TiTokDecoder
from .modules.maskgit_vqgan import Decoder as Pixel_Decoder
from .modules.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
from omegaconf import OmegaConf
from easydict import EasyDict as edict


class TiTok(nn.Module):
    def __init__(self):
        super().__init__()
        config = {
            "experiment": {
                "tokenizer_checkpoint": "tokenizer_titok_l32.bin",
                "generator_checkpoint": "generator_titok_l32.bin",
            },
            "model": {
                "vq_model": {
                    "codebook_size": 4096,
                    "token_size": 12,
                    "use_l2_norm": True,
                    "commitment_cost": 0.25,
                    "vit_enc_model_size": "large",
                    "vit_dec_model_size": "large",
                    "vit_enc_patch_size": 16,
                    "vit_dec_patch_size": 16,
                    "num_latent_tokens": 32,
                },
                "generator": {
                    "dropout": 0.1,
                    "attn_drop": 0.1,
                    "num_steps": 8,
                    "mask_schedule_strategy": "arccos",
                    "class_label_dropout": 0.1,
                    "image_seq_len": 32,
                    "condition_num_classes": 1000,
                },
            },
            "dataset": {"preprocessing": {"crop_size": 256}},
        }
        config = edict(config)
        self.config = config
        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config)

        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width**-0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width)
        )

        self.apply(self._init_weights)

        self.quantize = VectorQuantizer(
            codebook_size=config.model.vq_model.codebook_size,
            token_size=config.model.vq_model.token_size,
            commitment_cost=config.model.vq_model.commitment_cost,
            use_l2_norm=config.model.vq_model.use_l2_norm,
        )

        self.pixel_quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25
        )
        self.pixel_decoder = Pixel_Decoder(
            OmegaConf.create(
                {
                    "channel_mult": [1, 1, 2, 2, 4],
                    "num_resolutions": 5,
                    "dropout": 0.0,
                    "hidden_channels": 128,
                    "num_channels": 3,
                    "num_res_blocks": 2,
                    "resolution": 256,
                    "z_channels": 256,
                }
            )
        )
        
    def load_weights(self, model_path):
        g_p = os.path.join(model_path, 'generator_titok_l32.bin')
        t_p = os.path.join(model_path, 'tokenizer_titok_l32.bin')
        sd_g = torch.load(g_p, map_location="cpu")
        sd_t = torch.load(t_p, map_location="cpu")
        missing, unexpected = self.load_state_dict(sd_g, strict=False)
        missing, unexpected = self.load_state_dict(sd_t, strict=False)

    def _init_weights(self, module):
        """Initialize the weights.
        :param:
            module -> torch.nn.Module: module to initialize
        """
        if (
            isinstance(module, nn.Linear)
            or isinstance(module, nn.Conv1d)
            or isinstance(module, nn.Conv2d)
        ):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=0.02
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=0.02
            )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        if x.shape[-1] != self.config.dataset.preprocessing.crop_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(
                    self.config.dataset.preprocessing.crop_size,
                    self.config.dataset.preprocessing.crop_size,
                ),
                mode="bilinear",
                align_corners=False,
            )
            print(x.shape)
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        z_quantized, result_dict = self.quantize(z)
        return z_quantized, z, result_dict['min_encoding_indices']

    def decode(self, z_quantized):
        decoded_latent = self.decoder(z_quantized)
        quantized_states = torch.einsum(
            "nchw,cd->ndhw",
            decoded_latent.softmax(1),
            self.pixel_quantize.embedding.weight,
        )
        decoded = self.pixel_decoder(quantized_states)
        return decoded

    def decode_tokens(self, tokens):
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape  # B x N
        z_quantized = self.quantize.get_codebook_entry(tokens.reshape(-1)).reshape(
            batch, 1, seq_len, -1
        )
        if self.quantize.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        z_quantized = rearrange(z_quantized, "b h w c -> b c h w").contiguous()
        decoded = self.decode(z_quantized)
        return decoded


class VectorQuantizer(torch.nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        token_size: int = 256,
        commitment_cost: float = 0.25,
        use_l2_norm: bool = False,
    ):
        super().__init__()
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_l2_norm = use_l2_norm

    # Ensure quantization is performed using f32
    @autocast(enabled=False)
    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = rearrange(z, "b h w c -> (b h w) c")

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, embedding.T)
        )

        min_encoding_indices = torch.argmin(d, dim=1)  # num_ele
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean(
            (z_quantized.detach() - z) ** 2
        )
        codebook_loss = torch.mean((z_quantized - z.detach()) ** 2)

        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, "b h w c -> b c h w").contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices.view(
                z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3]
            ),
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum("bd,dn->bn", indices, self.embedding.weight)
        else:
            raise NotImplementedError
        return z_quantized
