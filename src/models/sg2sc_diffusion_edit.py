from typing import *
from torch import Tensor, LongTensor
import os

import torch
from torch import nn

from tqdm import tqdm
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from .networks import *

from src.utils.logger import StatsLogger
from lightning.pytorch import LightningModule
import open_clip
from .clip_encoders import CLIPTextEncoder, T5TextEncoder

class Sg2ScDiffusionEdit(LightningModule):
    def __init__(self, config, num_objs, num_preds, pred_save_folder=None) -> None:
        super().__init__()

        self.train_hypers = config['training']
        text_encoder = "t5" if "t5" in config["model"]["name"] else "clip"
        self.model = Sg2ScDiffusion(
            num_objs=num_objs,
            num_preds=num_preds,
            text_encoder=text_encoder
        )
        self.pred_save_folder = pred_save_folder

        print('use text as condition, and pretrained clip embedding')
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_hypers['optimizer']['lr'])
        return optimizer
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.model.compute_losses(batch)
        loss = sum(loss_dict.values())
        log_dict = {f"train/{k}": v for k, v in loss_dict.items()}
        log_dict['train_loss'] = loss
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch['instructions']))
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss_dict = self.model.compute_losses(batch)
        loss = sum(loss_dict.values())
        log_dict = {f"val/{k}": v for k, v in loss_dict.items()}
        log_dict['val_loss'] = loss
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch['instructions']))
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # assert self.pred_save_folder is not None, "Please provide a folder to save predictions"
        condition_params = batch['sources']
        instructions = batch['instructions']
        conditions = self.model.get_condition(condition_params, instructions)
        target_x = batch['targets']['objs']
        target_e = batch['targets']['edges']
        # target_o = batch['targets']['objfeat_vq_recon']
        target_o = batch['targets']['objfeat_vitg14_features']
        target_mask = batch['targets']['obj_masks']
        generated_boxes = self.model.generate_samples(target_x, target_e, target_o, target_mask, conditions, cfg_scale=1.2)

        all_data = []
        sources = batch['sources']
        targets = batch['targets']
        instructions = batch['instructions']

        for k, v in sources.items():
            if isinstance(v, Tensor):
                sources[k] = v.detach().cpu().numpy()
        for k, v in targets.items():
            if isinstance(v, Tensor):
                targets[k] = v.detach().cpu().numpy()
        generated_boxes = generated_boxes.detach().cpu().numpy()

        batch_size = len(instructions)
        for i in range(batch_size):
            save_uid = targets['room_uid'][i]
            source_i = {}
            target_i = {}
            generate_i = {}
            instruction_i = instructions[i]

            for k, v in sources.items():
                source_i[k] = v[i]
            
            for k, v in targets.items():
                target_i[k] = v[i]

            source_i = self.delete_empty_from_samples(source_i)
            generate_i = self.delete_empty_from_samples(target_i, new_boxes=generated_boxes[i])
            target_i = self.delete_empty_from_samples(target_i)

            data_i = [source_i, target_i, generate_i, instruction_i]
            if self.pred_save_folder is None:
                all_data.append(data_i)
                continue
            save_folder = os.path.join(self.pred_save_folder, f"{save_uid}")
            save_path = os.path.join(save_folder, "all_data.pt")
            os.makedirs(save_folder, exist_ok=True)
            torch.save(data_i, save_path)
        return all_data
    
    def delete_empty_from_samples(self, samples, new_boxes=None):
        mask = samples['obj_masks'] == 1
        if new_boxes is not None:
            boxes = new_boxes
        else:
            boxes = samples['boxes']
        
        boxes = boxes[mask]
        class_labels = samples['class_labels']
        objfeat_vq_recon = samples['objfeat_vq_recon'][mask]

        return {
            'translations': boxes[:, :3],
            'sizes': boxes[:, 3:6],
            'angles': boxes[:, 6:8],
            'class_labels': class_labels,
            'objfeat_vq_recon': objfeat_vq_recon,
            'jids': samples['jids'],
            'room_uid': samples['room_uid'],
        }
    
    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint["state_dict"].keys()):
            if "clip_model" in k:
                del checkpoint["state_dict"][k]
    
    def generate_samples(self, x, e, o, mask, source_params, instructions, cfg_scale=1.):
        text_last_hidden_state, text_embeds = self.model.get_text_condition(instructions, self.device)
        source_x = source_params["objs"]
        source_e = source_params["edges"]
        # source_o = source_params["objfeat_vq_recon"]
        source_o = source_params["objfeat_vitg14_features"]
        source_mask = source_params["obj_masks"]
        source_boxes = source_params["boxes"]

        bbox = self.model.generate_samples_concat(x, e, o, mask, source_x, source_e, source_o, source_mask, source_boxes, condition=text_last_hidden_state, global_condition=text_embeds, cfg_scale=cfg_scale)
        
        return bbox

class Sg2ScDiffusion(nn.Module):
    def __init__(self,
        num_objs: int, num_preds: int,
        diffusion_type="ddpm",
        cfg_drop_ratio=0.5,
        use_objfeat=True,
        text_encoder="t5"
    ):
        super().__init__()

        # TODO: make these parameters configurable
        if diffusion_type == "ddpm":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001, beta_end=0.02,
                beta_schedule="linear",
                variance_type="fixed_small",
                prediction_type="epsilon",
                clip_sample=True,
                clip_sample_range=1.
            )
        elif diffusion_type == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001, beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
                clip_sample=True,
                clip_sample_range=1.
            )
        else:
            raise NotImplementedError

        if text_encoder == "t5":
            self.clip_model = T5TextEncoder()
        elif text_encoder == "clip":
            self.clip_model = CLIPTextEncoder()

        # TODO: make these parameters configurable
        self.network = Sg2ScTransformerDiffusionWrapper(
            node_dim=num_objs+1,   # +1 for empty node (not really used)
            edge_dim=num_preds+1,  # +1 for empty edge
            t_dim=128, attn_dim=512,
            global_condition_dim=self.clip_model.text_emb_dim if text_encoder=="clip" else None,
            context_dim=self.clip_model.text_emb_dim,  # not use text condition in Sg2Sc
            n_heads=8, n_layers=5,
            gated_ff=True, dropout=0.1, ada_norm=True,
            cfg_drop_ratio=cfg_drop_ratio,
            use_objfeat=use_objfeat,
        )

        self.num_objs = num_objs
        self.num_preds = num_preds
        self.use_objfeat = use_objfeat

        self.cfg_scale = 1.  # for information logging

    def compute_losses(self, sample_params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        source_params = sample_params["sources"]
        target_params = sample_params["targets"]
        instructions = sample_params["instructions"]

        # source_condition, text_last_hidden_state, text_embeds = self.get_condition(source_params, instructions)
        # text_last_hidden_state, text_embeds = self.get_text_condition(instructions)
        # Unpack sample params
        x = target_params["objs"]          # (B, N)
        e = target_params["edges"]         # (B, N, N)
        o = target_params["objfeat_vq_recon"]  # (B, N, K)
        mask = target_params["obj_masks"]  # (B, N)
        boxes = target_params["boxes"]     # (B, N, 8)

        noise = torch.randn_like(boxes)

        B, device = x.shape[0], x.device

        timesteps = torch.randint(1, self.scheduler.config.num_train_timesteps, (B,)).to(device)

        # Mask out the padding boxes
        box_mask = mask.unsqueeze(-1)  # (B, N, 1)
        noise = noise * box_mask
        boxes = boxes * box_mask

        target = noise
        noisy_boxes = self.scheduler.add_noise(boxes, noise, timesteps) * box_mask

        source_x = source_params["objs"]          # (B, N)
        source_e = source_params["edges"]         # (B, N, N)
        source_o = source_params["objfeat_vq_recon"]  # (B, N, K)
        # source_o = source_params["objfeat_vq_indices"]/64  # (B, N, K)
        source_mask = source_params["obj_masks"]  # (B, N)
        source_boxes = source_params["boxes"]     # (B, N, 8)

        original_num = x.shape[1]
        new_e = torch.ones((B, 2*original_num, 2*original_num), device=device).long() * self.num_preds
        new_e[:, :original_num, :original_num] = e
        new_e[:, original_num:, original_num:] = source_e

        noisy_boxes = torch.cat([noisy_boxes, source_boxes], dim=1)
        x = torch.cat([x, source_x], dim=1)
        o = torch.cat([o, source_o], dim=1)
        new_mask = torch.cat([mask, source_mask], dim=1)
        emtpy_mask = torch.zeros_like(source_mask)
        new_box_mask = torch.cat([mask, emtpy_mask], dim=1).unsqueeze(-1)

        text_last_hidden_state, text_embeds = self.get_text_condition(instructions, device)

        pred = self.network(
            noisy_boxes, x, new_e, o,  # `x`, `e` and `o` as conditions
            timesteps, mask=new_mask, condition=text_last_hidden_state, global_condition=text_embeds, oringial_obj_num=original_num
        ) * new_box_mask

        losses = {}
        losses["pos_mse"] = F.mse_loss(pred[:, :original_num, :3], target[..., :3], reduction="none").sum() / box_mask.sum()
        losses["size_mse"] = F.mse_loss(pred[:, :original_num, 3:6], target[..., 3:6], reduction="none").sum() / box_mask.sum()
        losses["angle_mse"] = F.mse_loss(pred[:, :original_num, 6:8], target[..., 6:8], reduction="none").sum() / box_mask.sum()

        # try:
        #     for k, v in losses.items():
        #         StatsLogger.instance()[k].update(v.item() * B, B)
        # except:  # `StatsLogger` is not initialized
        #     pass
        return losses
    
    def get_condition(self, source_params, instructions: List[str]) -> Tensor:
        source_x = source_params["objs"]          # (B, N)
        source_e = source_params["edges"]         # (B, N, N)
        # source_o = source_params["objfeat_vq_recon"]  # (B, N, K)
        source_o = source_params["objfeat_vitg14_features"]
        source_mask = source_params["obj_masks"]  # (B, N)
        source_boxes = source_params["boxes"]     # (B, N, 8)
        device = source_x.device

        text_last_hidden_state, text_embeds = self.clip_model(instructions, device=device)

        # source_condition = self.condition_network(source_boxes, source_x, source_e, source_o, condition=condition_cross, mask=source_mask)
        source_condition = self.condition_network(source_boxes, source_x, source_e, source_o, mask=source_mask)
        # source_condition = self.condition_network(source_boxes, source_x, source_e, source_o, mask=source_mask, condition=text_last_hidden_state)
        # source_condition = source_condition*source_mask.unsqueeze(-1)
        source_condition = torch.cat([source_condition, text_last_hidden_state], dim=1)

        return source_condition, text_last_hidden_state, text_embeds
    
    def get_text_condition(self, instructions: List[str], device) -> Tensor:
        text_last_hidden_state, text_embeds = self.clip_model(instructions, device=device)
        return text_last_hidden_state, text_embeds

    @torch.no_grad()
    def generate_samples_concat(self,
        x: LongTensor, e: LongTensor, o: Optional[LongTensor], mask: LongTensor,
        source_x: LongTensor, source_e: LongTensor, source_o: Optional[LongTensor], source_mask: LongTensor, source_boxes: Tensor,
        condition: Optional[Tensor]=None,
        global_condition: Optional[Tensor]=None,
        vqvae_model: nn.Module=None,
        num_timesteps: Optional[int]=100,
        cfg_scale=1.
    ):
        self.cfg_scale = cfg_scale
        B, N, device = x.shape[0], x.shape[1], x.device

        if self.use_objfeat and vqvae_model is not None:
            with torch.no_grad():
                # Decode objfeat indices to objfeat embeddings
                B, N = o.shape[:2]
                o = vqvae_model.reconstruct_from_indices(
                    o.reshape(B*N, -1)
                ).reshape(B, N, -1)

        x = torch.cat([x, source_x], dim=1)
        new_e = torch.ones((B, 2*N, 2*N), device=device).long() * self.num_preds
        new_e[:, :N, :N] = e
        new_e[:, N:, N:] = source_e
        o = torch.cat([o, source_o], dim=1)
        new_mask = torch.cat([mask, source_mask], dim=1)

        boxes = torch.randn(B, N, 8).to(device)

        # Mask out the padding boxes
        box_mask = mask.unsqueeze(-1)  # (B, N, 1)
        boxes = boxes * box_mask

        if num_timesteps is None:
            num_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(num_timesteps)
        # stop_index = self.scheduler.timesteps[int(len(self.scheduler.timesteps)*0.95)]
        for t in tqdm(self.scheduler.timesteps, desc="Generating scenes", ncols=125):
            input_boxes = torch.cat([boxes, source_boxes], dim=1)
            pred = self.network(input_boxes, x, new_e, o, t, condition=condition, mask=new_mask, cfg_scale=cfg_scale, global_condition=global_condition, oringial_obj_num=N)
            pred = pred[:, :N] * box_mask
            boxes = self.scheduler.step(pred, t, boxes).prev_sample * box_mask
            # if t == stop_index:
            #     break

        return boxes


class Sg2ScTransformerDiffusionWrapper(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int,
        attn_dim=512, t_dim=128,
        global_dim: Optional[int]=None,
        global_condition_dim: Optional[int]=None,
        context_dim: Optional[int]=None,
        n_heads=8, n_layers=5,
        gated_ff=True, dropout=0.1, ada_norm=True,
        cfg_drop_ratio=0.2,
        use_objfeat=True
    ):
        super().__init__()

        if not ada_norm:
            global_dim = t_dim  # not use AdaLN, use global information in graph-attn instead

        self.node_embed = nn.Sequential(
            nn.Embedding(node_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, attn_dim),
        )
        if use_objfeat:
            self.node_proj_in = nn.Linear(attn_dim+1280+8, attn_dim)  # TODO: make `1280` configurable
        else:
            self.node_proj_in = nn.Linear(attn_dim+8, attn_dim)

        self.edge_embed = nn.Sequential(
            nn.Embedding(edge_dim, attn_dim//4),  # TODO: make `//4` configurable
            nn.GELU(),
            nn.Linear(attn_dim//4, attn_dim//4),
        )

        self.time_embed = nn.Sequential(
            Timestep(t_dim),
            TimestepEmbed(t_dim, t_dim)
        )

        if global_condition_dim is not None:
            self.global_condition_embed = nn.Sequential(
                nn.Linear(global_condition_dim, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, context_dim)
            )

        self.transformer_blocks = nn.ModuleList([
            GraphTransformerBlock(
                attn_dim, attn_dim//4, attn_dim, global_dim,
                context_dim, t_dim,
                n_heads, gated_ff, dropout, ada_norm
            ) for _ in range(n_layers)
        ])

        self.proj_out = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, 8)
        )

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.use_global_info = global_dim is not None
        self.cfg_drop_ratio = cfg_drop_ratio

    def forward(self,
        box: Tensor, x: LongTensor, e: LongTensor, o: Optional[Tensor],
        t: LongTensor, global_condition: Optional[Tensor]=None,
        condition: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, condition_mask: Optional[LongTensor]=None,
        cfg_scale=1.,
        oringial_obj_num=None,
    ):
        if not torch.is_tensor(t):
            if isinstance(t, (int, float)):  # single timestep
                t = torch.tensor([t], device=x.device)
            else:  # list of timesteps
                assert len(t) == x.shape[0]
                t = torch.tensor(t, device=x.device)
        else:  # is tensor
            if t.dim() == 0:
                t = t.unsqueeze(-1).to(x.device)
        # Broadcast to batch dimension, in a way that's campatible with ONNX/Core ML
        t = t * torch.ones(x.shape[0], dtype=t.dtype, device=t.device)

        x_emb = self.node_embed(x)
        if o is not None:
            x_emb = self.node_proj_in(torch.cat([x_emb, o, box], dim=-1))
        else:
            x_emb = self.node_proj_in(torch.cat([x_emb, box], dim=-1))
        e_emb = self.edge_embed(e)
        t_emb = self.time_embed(t)
        if self.use_global_info:
            y_emb = t_emb
        else:
            y_emb =None

        # Mask out the diagonal
        eye_mask = torch.eye(x.shape[1], device=x.device).bool().unsqueeze(0)  # (1, N, N)
        e_emb = e_emb * (~eye_mask).float().unsqueeze(-1)

        # Instance embeddings (TODO: do we need this for Sg2Sc model?)
        # inst_emb = get_1d_sincos_encode(
        #     torch.arange(x_emb.shape[1], device=x_emb.device),
        #     x_emb.shape[-1], x_emb.shape[1]
        # ).unsqueeze(0)  # (1, n, dx)
        # x_emb = x_emb + inst_emb

        # Classifier-free guidance in training
        if self.training and self.cfg_drop_ratio > 0.:
            assert cfg_scale == 1., "Do not use `cfg_scale` during training"
            # empty_e_emb = torch.zeros_like(e_emb[0]).unsqueeze(0)
            # empty_prob = torch.rand(e_emb.shape[0], device=e_emb.device) < self.cfg_drop_ratio
            # e_emb[empty_prob, ...] = empty_e_emb

            if oringial_obj_num is not None:
                empty_source_box_emb = torch.zeros_like(x_emb[0:1, oringial_obj_num:, -8:])
                empty_prob = torch.rand(x_emb.shape[0], device=x.device) < self.cfg_drop_ratio
                x_emb[empty_prob, oringial_obj_num:, -8:] = empty_source_box_emb

            empty_condition = torch.zeros_like(condition[0]).unsqueeze(0)  # e.g., for CLIP ViT-B/32: (1, 77, 512)
            empty_prob = torch.rand(condition.shape[0], device=x.device) < self.cfg_drop_ratio
            condition[empty_prob, ...] = empty_condition
            if global_condition is not None:
                empty_embeds = torch.zeros_like(global_condition[0]).unsqueeze(0)  # e.g., for CLIP ViT-B/32: (1, 512)
                empty_prob = torch.rand(global_condition.shape[0], device=x.device) < self.cfg_drop_ratio
                global_condition[empty_prob, ...] = empty_embeds
            
            inst_emb = get_1d_sincos_encode(
                torch.arange(x_emb.shape[1], device=x_emb.device),
                x_emb.shape[-1], x_emb.shape[1]
            ).unsqueeze(0)  # (1, n, dx)
            x_emb = x_emb + inst_emb
        elif not self.training and cfg_scale == 1.:
            inst_emb = get_1d_sincos_encode(
                torch.arange(x_emb.shape[1], device=x_emb.device),
                x_emb.shape[-1], x_emb.shape[1]
            ).unsqueeze(0)  # (1, n, dx)
            x_emb = x_emb + inst_emb

        # Prepare for classifier-free guidance in inference
        if not self.training and cfg_scale != 1.:
            # empty_e_emb = torch.zeros_like(e_emb)
            # e_emb = torch.cat([empty_e_emb, e_emb], dim=0)
            # x_emb = torch.cat([x_emb, x_emb], dim=0)
            # y_emb = torch.cat([y_emb, y_emb], dim=0) if y_emb is not None else None
            # t_emb = torch.cat([t_emb, t_emb], dim=0)
            # if condition is not None:
            #     condition = torch.cat([condition, condition], dim=0)
            # if mask is not None:
            #     mask = torch.cat([mask, mask], dim=0)
            # if condition_mask is not None:
            #     condition_mask = torch.cat([condition_mask, condition_mask], dim=0)
            if oringial_obj_num is not None:
                empty_x_emb = x_emb.clone()
                empty_x_emb[..., oringial_obj_num:, -8:] = 0

            inst_emb = get_1d_sincos_encode(
                torch.arange(x_emb.shape[1], device=x_emb.device),
                x_emb.shape[-1], x_emb.shape[1]
            ).unsqueeze(0)  # (1, n, dx)
            x_emb = x_emb + inst_emb
            if oringial_obj_num is not None:
                empty_x_emb = empty_x_emb + inst_emb

            empty_condition = torch.zeros_like(condition)  # e.g., for CLIP ViT-B/32: (B, 77, 512)
            condition = torch.cat([empty_condition, condition], dim=0)
            x_emb = torch.cat([empty_x_emb, x_emb], dim=0)
            y_emb = torch.cat([y_emb, y_emb], dim=0) if y_emb is not None else None
            # empty_e_emb = torch.zeros_like(e_emb)
            e_emb = torch.cat([e_emb, e_emb], dim=0)
            t_emb = torch.cat([t_emb, t_emb], dim=0)
            if global_condition is not None:
                empty_global_condition = torch.zeros_like(global_condition)  # e.g., for CLIP ViT-B/32: (B, 512)
                global_condition = torch.cat([empty_global_condition, global_condition], dim=0)
            
            if mask is not None:
                mask = torch.cat([mask, mask], dim=0)
            if condition_mask is not None:
                condition_mask = torch.cat([condition_mask, condition_mask], dim=0)
        
        if global_condition is not None:
            # t_emb += self.global_condition_embed(global_condition)
            condition = torch.cat([
                    condition,
                    self.global_condition_embed(global_condition).unsqueeze(1)
                ], dim=1)

        for block in self.transformer_blocks:
            x_emb, e_emb, y_emb = block(x_emb, e_emb, y_emb, t_emb, condition, mask, condition_mask)

        out_box = self.proj_out(x_emb)
        if mask is not None:
            out_box = out_box * mask.unsqueeze(-1)

        # Do classifier-free guidance in inference
        if not self.training and cfg_scale != 1.:
            out_box_uncond, out_box_cond = out_box.chunk(2, dim=0)
            out_box = out_box_uncond + cfg_scale * (out_box_cond - out_box_uncond)

        return out_box

class Sg2ScTransformerConditionWrapper(nn.Module):
    def __init__(self,
        node_dim: int, edge_dim: int,
        objfeat_dim: int=1280,
        bbox_dim: int=8,
        attn_dim=512, t_dim=128,
        global_dim: Optional[int]=None,
        global_condition_dim: Optional[int]=None,
        context_dim: Optional[int]=None,
        n_heads=8, n_layers=5,
        gated_ff=True, dropout=0.1, ada_norm=False,
        cfg_drop_ratio=0.2,
        use_objfeat=True
    ):
        super().__init__()

        self.node_embed = nn.Sequential(
            nn.Embedding(node_dim, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, attn_dim),
        )

        if not use_objfeat:
            objfeat_dim = 0
        self.objfeat_dim = objfeat_dim
        self.bbox_dim = bbox_dim
        node_feat_dim = attn_dim+objfeat_dim+bbox_dim
        self.node_proj_in = nn.Linear(node_feat_dim, attn_dim)
        # if use_objfeat:
        #     self.node_proj_in = nn.Linear(attn_dim+objfeat_dim+8, attn_dim)  # TODO: make `1280` configurable
        # else:
        #     self.node_proj_in = nn.Linear(attn_dim+8, attn_dim)

        self.edge_embed = nn.Sequential(
            nn.Embedding(edge_dim, attn_dim//4),  # TODO: make `//4` configurable
            nn.GELU(),
            nn.Linear(attn_dim//4, attn_dim//4),
        )

        self.transformer_blocks = nn.ModuleList([
            GraphTransformerBlock(
                attn_dim, attn_dim//4, attn_dim, global_dim,
                context_dim, t_dim,
                n_heads, gated_ff, dropout, ada_norm
            ) for _ in range(n_layers)
        ])

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.use_global_info = global_dim is not None
        self.cfg_drop_ratio = cfg_drop_ratio

    def forward(self,
        box: Tensor, x: LongTensor, e: LongTensor, o: Optional[Tensor],
        global_condition: Optional[Tensor]=None,
        condition: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, condition_mask: Optional[LongTensor]=None,
        cfg_scale=1.
    ):
        x_emb = self.node_embed(x)
        # if o is not None:
        #     x_emb = self.node_proj_in(torch.cat([x_emb, o, box], dim=-1))
        # else:
        #     x_emb = self.node_proj_in(torch.cat([x_emb, box], dim=-1))
        x_cat = [x_emb]
        if self.objfeat_dim > 0:
            x_cat.append(o)
        if self.bbox_dim > 0:
            x_cat.append(box)
        x_emb = self.node_proj_in(torch.cat(x_cat, dim=-1))
        e_emb = self.edge_embed(e)
        y_emb =None

        # Mask out the diagonal
        eye_mask = torch.eye(x.shape[1], device=x.device).bool().unsqueeze(0)  # (1, N, N)
        e_emb = e_emb * (~eye_mask).float().unsqueeze(-1)

        # Instance embeddings (TODO: do we need this for Sg2Sc model?)
        inst_emb = get_1d_sincos_encode(
            torch.arange(x_emb.shape[1], device=x_emb.device),
            x_emb.shape[-1], x_emb.shape[1]
        ).unsqueeze(0)  # (1, n, dx)
        x_emb = x_emb + inst_emb

        for block in self.transformer_blocks:
            x_emb, e_emb, y_emb = block(x_emb, e_emb, y_emb, None, condition, mask, condition_mask)

        return x_emb