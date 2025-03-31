import os
import pickle
import torch
import json
from copy import deepcopy
import numpy as np
from PIL import Image
from torchvision.transforms import PILToTensor

from .sg2sc_diffusion_edit import Sg2ScDiffusionEdit
from .sg_diffusion_vq_objfeat_edit import SgObjfeatVQDiffusionEdit
from .objfeat_vqvae import ObjectFeatureVQVAE
from lightning.pytorch import LightningModule
from src.utils.util import construct_scene_from_vq_objdata, render_generated_scene, get_blender_render, iou_3d

from src.models.clip_encoders import CLIPImageEncoder
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from sentence_transformers import SentenceTransformer

from constants import EDITROOM_DATA_FOLDER

class RoomEdit(LightningModule):
    def __init__(self, sg_config, sg2sc_config, num_objs, num_preds, pred_save_folder=None, 
                 postprocess_func=None, object_dataset=None, raw_dataset=None, all_classes=None) -> None:
        super().__init__()
        self.num_objs = num_objs
        self.num_preds = num_preds
        self.pred_save_folder = pred_save_folder
        self.max_length = sg_config['data']['max_length']

        sg_load_path = sg_config['load_path']
        sg2sc_load_path = sg2sc_config['load_path']

        self.sg_model = SgObjfeatVQDiffusionEdit.load_from_checkpoint(sg_load_path, strict=False, map_location="cpu", config=sg_config, num_objs=num_objs, num_preds=num_preds)
        self.sg2sc_model = Sg2ScDiffusionEdit.load_from_checkpoint(sg2sc_load_path, strict=False, map_location="cpu", config=sg2sc_config, num_objs=num_objs, num_preds=num_preds)
        print("Models loaded successfully")

        model_folder = os.path.join(EDITROOM_DATA_FOLDER, "objfeat_vqvae")
        with open(os.path.join(model_folder, "objfeat_bounds.pkl"), "rb") as f:
            kwargs = pickle.load(f)
        self.vqvae_model = ObjectFeatureVQVAE("openshape_vitg14", "gumbel", **kwargs)
        ckpt_path = f"{model_folder}/threedfront_objfeat_vqvae_epoch_01999.pth"
        self.vqvae_model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
        self.vqvae_model.eval()

        self.postprocess_func = postprocess_func
        self.object_dataset = object_dataset
        self.raw_dataset = raw_dataset
        self.all_classes = all_classes

        self.desc_emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.clip_image = CLIPImageEncoder()
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # assert self.pred_save_folder is not None, "Please provide a folder to save predictions"
        source_params = batch['sources']
        instructions = batch['instructions']

        turn = 0
        condition_params = {}
        condition_params['objs'] = source_params['objs'].clone()
        condition_params['boxes'] = source_params['boxes'].clone()
        condition_params['edges'] = source_params['edges'].clone()
        condition_params['obj_masks'] = source_params['obj_masks'].clone()
        condition_params['objfeat_vq_recon'] = source_params['objfeat_vq_recon'].clone()
        condition_params['objfeat_vq_indices'] = source_params['objfeat_vq_indices'].clone()
        condition_params['objfeat_vitg14_features'] = source_params['objfeat_vitg14_features'].clone()

        finished_index = {}
        while True:
            current_index = []
            new_instructions = []
            for i, ins in enumerate(instructions):
                if type(ins) == str:
                    finished_index[i] = "finished"
                elif type(ins) == list:
                    if turn < len(ins)-1:
                        new_instructions.append(ins[turn])
                        current_index.append(i)
                        finished_index[i] = "unfinished"
                    else:
                        finished_index[i] = "finished"

            is_final_turn = len(new_instructions) == 0
            if is_final_turn:
                for ins in instructions:
                    if type(ins) == list:
                        new_instructions.append(ins[-1])
                    elif type(ins) == str:
                        new_instructions.append(ins)

            if is_final_turn:
                generated_boxes, generated_mask, generated_x, generated_e, generated_o_recon, generated_o, generated_o_vitg14_features = self.generate_single(condition_params, new_instructions)
                break
            else:
                new_condition_params = {}
                new_condition_params['objs'] = condition_params['objs'][current_index]
                new_condition_params['boxes'] = condition_params['boxes'][current_index]
                new_condition_params['edges'] = condition_params['edges'][current_index]
                new_condition_params['obj_masks'] = condition_params['obj_masks'][current_index]
                new_condition_params['objfeat_vq_recon'] = condition_params['objfeat_vq_recon'][current_index]
                new_condition_params['objfeat_vq_indices'] = condition_params['objfeat_vq_indices'][current_index]
                new_condition_params['objfeat_vitg14_features'] = condition_params['objfeat_vitg14_features'][current_index]

                generated_boxes, generated_mask, generated_x, generated_e, generated_o_recon, generated_o, generated_o_vitg14_features = self.generate_single(new_condition_params, new_instructions)

                condition_params['objs'][current_index] = generated_x
                condition_params['boxes'][current_index] = generated_boxes
                condition_params['edges'][current_index] = generated_e
                condition_params['obj_masks'][current_index] = generated_mask
                condition_params['objfeat_vq_recon'][current_index] = generated_o_recon
                condition_params['objfeat_vq_indices'][current_index] = generated_o
                condition_params['objfeat_vitg14_features'][current_index] = generated_o_vitg14_features

                turn += 1

        all_data = []
        sources = batch['sources']
        targets = batch['targets']
        instructions = batch['instructions']

        for k, v in sources.items():
            if isinstance(v, torch.Tensor):
                sources[k] = v.detach().cpu().numpy()
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                targets[k] = v.detach().cpu().numpy()
                
        generated_boxes = generated_boxes.detach().cpu().numpy()
        generated_mask = generated_mask.detach().cpu().numpy()
        generated_x = generated_x.detach().cpu().numpy()
        generated_o_recon = generated_o_recon.detach().cpu().numpy()
        generated_o = generated_o.detach().cpu().numpy()

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

            generate_i = {
                "boxes": generated_boxes[i],
                "obj_masks": generated_mask[i],
                "objs": generated_x[i],
                "objfeat_vq_recon": generated_o_recon[i],
                "objfeat_vq_indices": generated_o[i],
                "room_uid": save_uid,
            }
            source_i = self.delete_empty_from_samples(source_i)
            generate_i = self.delete_empty_from_samples(generate_i)
            target_i = self.delete_empty_from_samples(target_i)

            data_i = [source_i, target_i, generate_i, instruction_i]
            if self.pred_save_folder is None:
                all_data.append(data_i)
                continue
            save_folder = os.path.join(self.pred_save_folder, f"{save_uid}")
            save_path = os.path.join(save_folder, "all_data.pt")
            os.makedirs(save_folder, exist_ok=True)
            torch.save(data_i, save_path)
            self.postprocess_generation(data_i, save_folder)
        return all_data
    
    def generate_single(self, condition_params, instructions):
        batch_size = len(instructions)
        generated_x, generated_e, generated_token_o = self.sg_model.generate_samples(batch_size, self.max_length, condition_params, instructions, cfg_scale=1.5, source_cfg_scale=2.0)
        generated_x, generated_mask, generated_e, generated_o = self.sg_model.post_process(generated_x, generated_e, generated_token_o)

        B, N = generated_o.shape[:2]
        generated_o_recon = self.vqvae_model.reconstruct_from_indices(generated_o.reshape(B*N, -1)).reshape(B, N, -1)
        generated_o_recon = generated_o_recon*generated_mask.unsqueeze(-1)
        generated_o_recon = self.process_with_source(generated_o_recon, condition_params['objfeat_vq_recon'], generated_mask, condition_params['obj_masks'], instructions)

        generated_o_vitg14_features = self.get_cloest_furniture(generated_mask, generated_o_recon, generated_x)

        generated_boxes = self.sg2sc_model.generate_samples(generated_x, generated_e, generated_o_vitg14_features, generated_mask, condition_params, instructions, cfg_scale=1.2)
        generated_o = generated_o*generated_mask.unsqueeze(-1) + torch.ones_like(generated_o)*(1-generated_mask.unsqueeze(-1))*64
        return generated_boxes, generated_mask, generated_x, generated_e, generated_o_recon, generated_o, generated_o_vitg14_features

    def delete_empty_from_samples(self, samples, new_boxes=None):
        mask = samples['obj_masks'] == 1
        if new_boxes is not None:
            boxes = new_boxes
        else:
            boxes = samples['boxes']
        
        boxes = boxes[mask]
        objs = samples['objs'][mask]
        objfeat_vq_recon = samples['objfeat_vq_recon'][mask]
        objfeat_vq_indices = samples['objfeat_vq_indices'][mask]

        return_dict = {
            'translations': boxes[:, :3],
            'sizes': boxes[:, 3:6],
            'angles': boxes[:, 6:8],
            'objs': objs,
            'objfeat_vq_recon': objfeat_vq_recon,
            'objfeat_vq_indices': objfeat_vq_indices,
            'room_uid': samples['room_uid'],
        }
        if 'jids' in samples.keys():
            return_dict['jids']=samples['jids']
        
        return return_dict
    
    def process_with_source(self, generate_recon, source_recon, generated_mask, source_mask, instructions):

        norm_generate_recon = torch.nn.functional.normalize(generate_recon, p=2, dim=-1)
        norm_source_recon = torch.nn.functional.normalize(source_recon, p=2, dim=-1)

        # find the nearest neighbor for generated params to source params
        for i in range(len(generate_recon)):
            instruction = instructions[i]
            if "replace" in instruction:
                margin = 0.95
            else:
                margin = 0.8

            generated_mask_i = generated_mask[i]==1
            generate_recon_i = norm_generate_recon[i][generated_mask_i]

            source_mask_i = source_mask[i]==1
            source_recon_i = norm_source_recon[i][source_mask_i]

            orig_generate_recon_i = generate_recon[i][generated_mask_i]
            orig_source_recon_i = source_recon[i][source_mask_i]

            nn_pair_index = {}
            score = torch.zeros((len(generate_recon_i), len(source_recon_i)), device=generate_recon_i.device)
            for k in range(len(generate_recon_i)):
                for m in range(len(source_recon_i)):
                    total_score = torch.dot(generate_recon_i[k], source_recon_i[m])
                    total_score = total_score if total_score > margin else -torch.inf
                    score[k, m] = total_score
            while not torch.isneginf(score).all():
                row, col = torch.where(score == torch.max(score))
                row, col = row[0], col[0]
                nn_pair_index[row] = col
                score[row] = -torch.inf
                score[:, col] = -torch.inf

            for k in range(len(generate_recon_i)):
                if k in nn_pair_index.keys():
                    orig_generate_recon_i[k] = orig_source_recon_i[nn_pair_index[k]]

            generate_recon[i][generated_mask_i] = orig_generate_recon_i

        return generate_recon
    
    def get_cloest_furniture(self, generated_mask, generate_recon, generated_x):
        generated_o_vitg14_features = torch.zeros_like(generate_recon)
        for b in range(len(generated_mask)):
            for i in range(len(generated_mask[b])):
                if generated_mask[b][i] == 1:
                    objfeat_vq_recon = generate_recon[b][i].detach().cpu().numpy()
                    object_class = generated_x[b][i].item()
                    object_class = self.all_classes[object_class]
                    obj = self.object_dataset.get_closest_furniture_to_objfeat(object_class, objfeat_vq_recon)
                    objfeat_vitg14_features = obj.openshape_vitg14_features
                    objfeat_vitg14_features = torch.tensor(objfeat_vitg14_features).to(self.device)
                    generated_o_vitg14_features[b][i] = objfeat_vitg14_features
        return generated_o_vitg14_features
                    
    def postprocess_generation(self, data, save_folder):
        
        source_params, target_params, generate_params, instructions = data


        target_params = self.postprocess_func(target_params)
        target_scene, target_scene_trimesh = construct_scene_from_vq_objdata(target_params, self.object_dataset, all_classes=self.all_classes)

        generate_params = self.postprocess_func(generate_params)
        generate_scene, generate_scene_trimesh = construct_scene_from_vq_objdata(generate_params, self.object_dataset, all_classes=self.all_classes)

        iou_scores = self.calculate_iou(target_scene, generate_scene)

        target_image_folder = os.path.join(save_folder, "target")
        generate_image_folder = os.path.join(save_folder, "generate")
        try:
            target, target_images, scene_centroid, radius = render_generated_scene(target_scene_trimesh, output_folder=target_image_folder)
            pred, pred_images, _, _ = render_generated_scene(generate_scene_trimesh, scene_centroid=scene_centroid, radius=radius, output_folder=generate_image_folder)
        except Exception as e:
            print(f"Error in rendering scene using pyrender: {e}. Using blender instead.")
            source_index = self.raw_dataset.uid_to_scene_index[source_params['room_uid']]
            raw_source_scene = self.raw_dataset[source_index]

            get_blender_render(raw_source_scene, target_scene_trimesh, save_folder=target_image_folder, verbose=False, remove_mesh=True)
            get_blender_render(raw_source_scene, generate_scene_trimesh, save_folder=generate_image_folder, verbose=False, remove_mesh=True)

        image_similarity_scores = self.calculate_image_similarity(target_image_folder, generate_image_folder)
        iou_scores.update(image_similarity_scores)

        instruction_save_path = os.path.join(save_folder, "instruction.json")
        with open(instruction_save_path, "w") as f:
            f.write(json.dumps(instructions))
        
        score_save_path = os.path.join(save_folder, "scores.json")
        with open(score_save_path, "w") as f:
            f.write(json.dumps(iou_scores))
        return iou_scores

    def calculate_iou(self, scene1, scene2):
        def desc_similarity_score(desc_emb_model, text1, text2):
            emb1 = desc_emb_model.encode(text1)
            emb2 = desc_emb_model.encode(text2)
            return torch.cosine_similarity(torch.tensor(emb1), torch.tensor(emb2), dim=0)
        nn_pair_index = {}
        score = np.zeros((len(scene1), len(scene2)))
        for i in range(len(scene1)):
            for j in range(len(scene2)):
                score[i, j] = iou_3d(scene1[i], scene2[j])
        while not np.isneginf(score).all():
            row, col = np.where(score == np.max(score))
            row, col = row[0], col[0]
            nn_pair_index[row] = col
            score[row] = -np.inf
            score[:, col] = -np.inf
        for i in range(len(scene1)):
            if i not in nn_pair_index.keys():
                nn_pair_index[i] = 'not paired'
        score_iou_paired, score_class_paired = [], []
        for i in range(len(scene1)):
            if nn_pair_index[i] != 'not paired':
                iou = iou_3d(scene1[i], scene2[nn_pair_index[i]])
                class_sim = desc_similarity_score(self.desc_emb_model, scene1[i]['obj'].description(), scene2[nn_pair_index[i]]['obj'].description()).clamp(0, 1).item()
                class_iou = iou * class_sim
                score_iou_paired.append(iou)
                score_class_paired.append(class_iou)
            else:
                score_iou_paired.append(0)
                score_class_paired.append(0)

        scores = {
            'iou': float(np.mean(score_iou_paired)),
            'class_iou': float(np.mean(score_class_paired))
        }
        return scores
    
    def calculate_image_similarity(self, target_image_folder, pred_image_folder):
        totensor = PILToTensor()

        target_image_files = os.listdir(target_image_folder)
        pred_image_files = os.listdir(pred_image_folder)
        assert len(target_image_files) == len(pred_image_files), "Number of images in target and predicted folders do not match"
        target_image_files.sort()
        pred_image_files.sort()

        target_images, pred_images = [], []
        for target_image_file, pred_image_file in zip(target_image_files, pred_image_files):
            target_image_path = os.path.join(target_image_folder, target_image_file)
            pred_image_path = os.path.join(pred_image_folder, pred_image_file)

            target_image = Image.open(target_image_path).convert("RGB")
            pred_image = Image.open(pred_image_path).convert("RGB")

            target_images.append(target_image)
            pred_images.append(pred_image)

        clip_pred = self.clip_image(pred_images)
        clip_target = self.clip_image(target_images)
        clip_score = torch.nn.functional.cosine_similarity(clip_pred, clip_target, dim=-1).mean()

        pred_images_tensor = torch.stack([totensor(image)/255 for image in pred_images]).to(self.device)
        target_images_tensor = torch.stack([totensor(image)/255 for image in target_images]).to(self.device)

        pred_images_tensor = pred_images_tensor*2 -1
        target_images_tensor = target_images_tensor*2 -1
        lpips_score = self.lpips(pred_images_tensor, target_images_tensor)

        scores = {
            'lpips': lpips_score.item(),
            'clip': clip_score.item()
        }
        return scores


        