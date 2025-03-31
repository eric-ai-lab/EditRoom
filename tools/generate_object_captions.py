import sys, os
# append parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import Dataset, DataLoader
from typing import Any, Optional, Dict, Sequence
import json
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import BasePredictionWriter

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import pickle

PRECISION = torch.bfloat16
MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
# MODEL_NAME ="llava-hf/llava-v1.6-vicuna-7b-hf"
OBJECT_FOLDER = '/data1/zhengkz/3D_datasets/3D-FRONT/3D-FUTURE-model'

class PointCloudDataset(Dataset):
    def __init__(self, all_objects):
        jids = []
        self.labels = []
        for obj in all_objects:
            jid = obj.model_jid
            if jid not in jids:
                image_path = obj.raw_model_path.replace("raw_model.obj", "image.jpg")
                if not os.path.exists(image_path):
                    print("Image not found for", jid)
                    continue
                jids.append(jid)
                self.labels.append(obj.label.replace("_", " "))

        self.data = list(jids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        jid = self.data[idx]
        label = self.labels[idx]
        image_path = os.path.join(OBJECT_FOLDER, jid, "image.jpg")
        image = Image.open(image_path)
        prompt = f"[INST] <image>\nUse a single sentence to describe the object shape and apperance (color, style, etc.). The current object category is: {label}. [/INST]"
        # prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nUse a single sentence to describe the object shape and apperance. The description should make the object distinguish than other same type objects. ASSISTANT:"
        
        inputs = {"prompt": prompt, "image": image}
        inputs['jid'] = jid

        return inputs
    
    @staticmethod
    def collate_fn(batch):
        images = [sample["image"] for sample in batch]
        prompts = [sample["prompt"] for sample in batch]
        jids = [sample["jid"] for sample in batch]
        return {"image": images, "prompt": prompts, "jids": jids}

class PredWriter(BasePredictionWriter):
    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Any,  # complex variables is ok
        batch_indices: list[list[list[int]]],
    ) -> None:
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, predictions)
        torch.distributed.barrier()
        if not trainer.is_global_zero:
            return
        predictions = sum(gathered, [])
        all_predictions = {}
        for pred in predictions:
            all_predictions.update(pred)
        with open("object_caption_mistral7B.json", "w") as f:
            json.dump(all_predictions, f)
        print('predictions saved')

class Captioner(LightningModule):
    def __init__(self):
        super().__init__()
        # self.processor = Blip2Processor.from_pretrained(MODEL_NAME)
        # self.model = Blip2ForConditionalGeneration.from_pretrained(MODEL_NAME).to(PRECISION)
        self.processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)
        # self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        self.model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        print("Model loaded") 
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs = self.processor(batch['prompt'], images=batch['image'], return_tensors="pt", padding=True).to(self.device)
        jids = batch['jids']
        output = self.model.generate(**inputs, max_new_tokens=100, pad_token_id=self.processor.tokenizer.eos_token_id, temperature=0.8, do_sample=True)
        captions = self.processor.batch_decode(output, skip_special_tokens=True, )
        object_dict = {}
        for jid, caption in zip(jids, captions):
            object_dict[jid] = [caption.split("[/INST]")[1].lower().replace("the object is", "").strip()]
            # object_dict[jid] = [caption]
        return object_dict

if __name__=="__main__":
    seed_everything(42)
    print("HF home:", os.environ["HF_HOME"])
    object_dataset_path1 = "/data1/zhengkz/3D_datasets/EditScene/threed_front_bedroom/threed_front_bedroom_objects.pkl"
    with open(object_dataset_path1, "rb") as f:
        object_dataset = pickle.load(f)
    object_dataset_path2 = "/data1/zhengkz/3D_datasets/EditScene/threed_front_diningroom/threed_front_diningroom_objects.pkl"
    with open(object_dataset_path2, "rb") as f:
        object_dataset.objects.extend(pickle.load(f).objects)
    object_dataset_path3 = "/data1/zhengkz/3D_datasets/EditScene/threed_front_livingroom/threed_front_livingroom_objects.pkl"
    with open(object_dataset_path3, "rb") as f:
        object_dataset.objects.extend(pickle.load(f).objects)
    dataset = PointCloudDataset(object_dataset.objects)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, 
                            collate_fn=dataset.collate_fn
                            )
    # encoder = ShapeEncoder()
    encoder = Captioner()
    pred_writer = PredWriter(write_interval="epoch")
    trainer = Trainer(accelerator="gpu", devices=8, callbacks=[pred_writer])
    trainer.predict(encoder, dataloader, return_predictions=False)
    
    # trainer.predict(encoder, dataloader, return_predictions=False)
    # make the process 0 exits only when all the processes have finished
    print("Done!")