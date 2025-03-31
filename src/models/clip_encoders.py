from typing import *
from PIL.Image import Image as PILImage

import torch
from torch import nn

from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers import AutoTokenizer, T5EncoderModel
import open_clip


class CLIPTextEncoder(nn.Module):
    def __init__(self,
        name="openai/clip-vit-base-patch32",
        # name="hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        max_length=77, device="cpu"
    ):
        super().__init__()
        self.name = name
        if "hf-hub" in name:
            self.text_encoder, _, _ = open_clip.create_model_and_transforms(name)
            del self.text_encoder.visual
            self.tokenizer = open_clip.get_tokenizer(name)
            self.text_encoder.text_pool_type = "none"
            self.text_emb_dim = 1280
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(name)
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(name).to(device).eval()

            self.text_emb_dim = self.text_encoder.config.hidden_size
            self.max_length = max_length
            self.device = device
            self.text_emb_dim = 512
            assert self.max_length == self.tokenizer.model_max_length
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, prompt: Union[str, List[str]], norm=True, device="cpu"):
        if "hf-hub" in self.name:
            tokenized = self.tokenizer(prompt).to(device)
            text_last_hidden_state = self.text_encoder.encode_text(tokenized)
            text_embeds = text_last_hidden_state[torch.arange(text_last_hidden_state.shape[0]), tokenized.argmax(dim=-1)]
            if norm:
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # L2 normalize
        else:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            text_encoder_output = self.text_encoder(
                text_input_ids.to(device)
            )

            text_last_hidden_state = text_encoder_output.last_hidden_state.float()  # (num_prompts, max_length, text_emb_dim)
            text_embeds = text_encoder_output.text_embeds.float()  # (num_prompts, text_emb_dim)
            if norm:
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # L2 normalize

        return text_last_hidden_state, text_embeds


class CLIPImageEncoder(nn.Module):
    def __init__(self,
        name="openai/clip-vit-base-patch32",
        device="cpu"
    ):
        super().__init__()

        self.image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(name).to(device).eval()

        self.image_emb_dim = self.image_encoder.config.hidden_size
        self.device = device

    @torch.no_grad()
    def forward(self, image: Union[PILImage, List[PILImage]], norm=True):
        device = self.image_encoder.device
        image = self.image_processor(images=image, return_tensors="pt").pixel_values.to(device)
        image_embeds = self.image_encoder(image).image_embeds.float()  # (num_images, image_emb_dim)
        if norm:
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        return image_embeds

class T5TextEncoder(nn.Module):
    def __init__(self, name="google-t5/t5-large"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.text_encoder = T5EncoderModel.from_pretrained(name).eval()

        self.text_emb_dim = self.text_encoder.config.d_model

        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, prompt: Union[str, List[str]], norm=True, device="cpu"):
        text_inputs = self.tokenizer(
            prompt,
            padding='max_length',  # Pad all sequences to the maximum length
            max_length=128,        # Maximum length to pad/truncate
            truncation=True,
            return_tensors="pt",
        ).to(device)
        # text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs['attention_mask']
        text_encoder_output = self.text_encoder(**text_inputs)

        text_last_hidden_state = text_encoder_output.last_hidden_state.float()  # (num_prompts, max_length, text_emb_dim)
        text_last_hidden_state = text_last_hidden_state * attention_mask.unsqueeze(-1).float()
        return text_last_hidden_state, None