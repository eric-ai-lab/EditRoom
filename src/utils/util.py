from typing import *
from argparse import Namespace
from torch.nn import Module
from torch.optim import Optimizer
from diffusers.training_utils import EMAModel
import shutil

import os
import json
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import torch
import numpy as np
import trimesh
from pathlib import Path
from src.data.threed_front_scene import orbit_camera
from PIL import Image
import pyrender
from tqdm import tqdm
from queue import Empty

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from sentence_transformers import SentenceTransformer, util
from shapely.geometry import Polygon
from src.models.clip_encoders import CLIPImageEncoder
from src.utils.visualize import export_scene, blender_render_scene

"""
# the command for running Xvfb
Xvfb :99 -screen 0 1960x1024x24 &
"""

FOVY = 60
RESOLUTION = 256



# Copied from https://github.com/huggingface/pytorch-image-models/timm/data/loader.py
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def yield_forever(iterator: Iterator[Any]):
    while True:
        for x in iterator:
            yield x


def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def save_experiment_params(args: Namespace, experiment_tag: str, directory: str) -> None:
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


def save_model_architecture(model: Module, directory: str) -> None:
    """Save the model architecture to a `.txt` file."""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    message = f'Number of trainable / all parameters: {num_trainable_params} / {num_params}\n\n' + str(model)

    with open(os.path.join(directory, 'model.txt'), 'w') as f:
        f.write(message)


def load_checkpoints(
    model: Module,
    ckpt_dir: str,
    ema_states: Optional[EMAModel]=None,
    optimizer: Optional[Optimizer]=None,
    epoch: Optional[int]=None,
    device=torch.device("cpu")
) -> int:
    """Load checkpoint from the given experiment directory and return the epoch of this checkpoint."""
    if epoch is not None and epoch < 0:
        epoch = None

    model_files = [f.split(".")[0] for f in os.listdir(ckpt_dir)
        if f.startswith("threedfront_objfeat_vqvae_epoch_") and f.endswith(".pth")]

    if len(model_files) == 0:  # no checkpoints found
        print(f"No checkpoint found in {ckpt_dir}, starting from scratch\n")
        return -1

    epoch = epoch or max([int(f[6:]) for f in model_files])  # load the latest checkpoint by default
    checkpoint_path = os.path.join(ckpt_dir, f"threedfront_objfeat_vqvae_epoch_{epoch:05d}.pth")
    if not os.path.exists(checkpoint_path):  # checkpoint file not found
        print(f"Checkpoint file {checkpoint_path} not found, starting from scratch\n")
        return -1

    print(f"Load checkpoint from {checkpoint_path}\n")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    if ema_states is not None:
        ema_states.load_state_dict(checkpoint["ema_states"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return epoch


def save_checkpoints(model: Module, optimizer: Optimizer, ckpt_dir: str, epoch: int, ema_states: Optional[EMAModel]=None) -> None:
    """Save checkpoint to the given experiment directory."""
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    if ema_states is not None:
        save_dict["ema_states"] = ema_states.state_dict()

    save_path = os.path.join(ckpt_dir, f"epoch_{epoch:05d}.pth")
    torch.save(save_dict, save_path)

"""
Code for Edit
"""

def process_with_source(generate_params, source_params, instruction):
    # if "Replace" in instruction:
    #     margin = 0.9
    # else:
    #     margin = 0
    margin = 0.95
    if "ulip_feats" not in generate_params:
        genertated_ulip = generate_params['objfeat_vq_recon']
        source_ulip = source_params['objfeat_vq_recon']
    else:
        genertated_ulip = generate_params['ulip_feats']
        source_ulip = source_params['ulip_feats']
    genertated_ulip = genertated_ulip / np.linalg.norm(genertated_ulip, axis=-1, keepdims=True)

    source_ulip = source_ulip / np.linalg.norm(source_ulip, axis=-1, keepdims=True)

    use_shape_feat = "objfeats" in generate_params and "objfeats" in source_params
    if use_shape_feat:
        genertated_shape = generate_params['objfeats']
        genertated_shape = genertated_shape / np.linalg.norm(genertated_shape, axis=-1, keepdims=True)
        source_shape = source_params['objfeats']
        source_shape = source_shape / np.linalg.norm(source_shape, axis=-1, keepdims=True)

    # find the nearest neighbor for generated params to source params
    # use cosine similarity of ulip_feats and objfeats_32

    nn_pair_index = {}
    score = np.zeros((len(genertated_ulip), len(source_ulip)))
    for i in range(len(genertated_ulip)):
        for j in range(len(source_ulip)):
            ulip_score = np.dot(genertated_ulip[i], source_ulip[j])
            if use_shape_feat:
                shape_score = np.dot(genertated_shape[i], source_shape[j])
                total_score = (ulip_score + shape_score)/2
            else:
                total_score = ulip_score
            total_score = total_score if total_score > margin else -np.inf
            score[i, j] = total_score
    while not np.isneginf(score).all():
        row, col = np.where(score == np.max(score))
        row, col = row[0], col[0]
        nn_pair_index[row] = col
        score[row] = -np.inf
        score[:, col] = -np.inf
    
    generated_jids = []
    source_jids = source_params['jids']
    for i in range(len(genertated_ulip)):
        if i in nn_pair_index.keys():
            generated_jids.append(source_jids[nn_pair_index[i]])
        else:
            generated_jids.append(None)
    
    generate_params['jids'] = generated_jids

    return generate_params

def construct_scene_from_vq_objdata(scene_params, object_dataset, all_classes=None):
    scene = []
    num_objects = len(scene_params['translations'])
    objs = scene_params['objs']
    objfeat_vq_recon = scene_params['objfeat_vq_recon']
    
    if 'jids' in scene_params:
        jids = scene_params['jids']
    else:
        jids = None

    for i in range(num_objects):
        obj = None
        if jids is not None:
            obj_jid = jids[i]
            if obj_jid is not None:
                # obj_jid = Path(obj_jid).parent.name
                obj = object_dataset.get_furniture_by_jid(obj_jid)
        if obj is None:
            object_class = all_classes[objs[i]]
            obj = object_dataset.get_closest_furniture_to_objfeat(object_class, objfeat_vq_recon[i])
        if obj is None:
            print(f"Could not find object for {object_class}")
            continue
        obj_path = obj.raw_model_path
        translate = scene_params['translations'][i]
        # translate = torch.tensor(translate, dtype=torch.float32, device=device)
        theta = -scene_params['angles'][i, 0]
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        # R = torch.tensor(R, dtype=torch.float32, device=device)
        # quaternion = matrix_to_quaternion(R)
        # quaternion[1] = -quaternion[1]
        # R = quaternion_to_matrix(quaternion)
        size = scene_params['sizes'][i]
        # size = torch.tensor(size, dtype=torch.float32, device=device)
        scene.append({
            'obj': obj,
            'translation': translate,
            'rotation': R,
            'size': size
        })

    scene_obs = []
    for config in scene:
        obj_path = config['obj'].raw_model_path
        translate = config['translation']
        R = config['rotation']
        size = config['size']

        mesh = trimesh.load(obj_path)
        # recentre the mesh
        bbox_aabb = mesh.bounding_box
        center_translation = -bbox_aabb.centroid
        mesh.apply_translation(center_translation)
        # scale the mesh
        scale = (size / bbox_aabb.extents)*2
        mesh.apply_scale(scale)

        transform = np.eye(4)
        transform[:3, :3] = R

        mesh.apply_transform(transform)
        mesh.apply_translation(translate)
        scene_obs.append(mesh)
    # scene_obs = trimesh.Scene(scene_obs)
    return scene, scene_obs

def render_generated_scene(trimesh_scene, output_folder=None, scene_centroid=None, radius=None, resolution=RESOLUTION, device=None, show_progress_bar=False):
    if type(trimesh_scene) == list:
        trimesh_scene = trimesh.Scene(trimesh_scene)
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
    if scene_centroid is None:
        scene_centroid = trimesh_scene.centroid
    trimesh_scene.apply_translation(-scene_centroid)

    if radius is None:
        radius = (np.sum(trimesh_scene.bounding_box.extents**2)**0.5 / 2)+2

    vers = [0] * 8 + [-30] * 8 + [-60] * 8 + [-89]
    hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0]
    poses = [orbit_camera(ver, hor, radius) for ver, hor in zip(vers, hors)]
    mesh_renderer = pyrender.OffscreenRenderer(resolution, resolution)
    pyrender_scene = pyrender.Scene.from_trimesh_scene(trimesh_scene, bg_color=(0.5, 0.5, 0.5, 1.0), ambient_light=(1.0, 1.0, 1.0))
    scene_obs = []
    scene_obs_images = []
    number = 0
    if show_progress_bar:
        poses = tqdm(poses, desc="Rendering")
    for pose in poses:
        img_white, _, _ = render_scene_seg(mesh_renderer, pyrender_scene, pose, need_seg=False)
        scene_obs.append(np.array(img_white))
        img = Image.fromarray(img_white)
        scene_obs_images.append(img)
        if output_folder is not None:
            img.save(f"{output_folder}/{number}.png")
        number += 1
    mesh_renderer.delete()
    if device is not None:
        scene_obs = torch.stack([torch.tensor(obs, dtype=torch.float32, device=device)/255 for obs in scene_obs]).permute(0, 3, 1, 2)
    return scene_obs, scene_obs_images, scene_centroid, radius

def render_scene_seg(mesh_renderer, scene, cam_pose, background_color=(1.0, 1.0, 1.0, 1.0), need_seg=True):
    if isinstance(scene, pyrender.Scene):
        pyrender_scene = scene
        pyrender_scene.bg_color = background_color
    else:
        pyrender_scene = pyrender.Scene.from_trimesh_scene(scene, ambient_light=(1.0, 1.0, 1.0), bg_color=background_color)
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(FOVY), aspectRatio=1.0, znear=0.01, zfar=100)
    cam_node = pyrender.Node(camera=camera, matrix=cam_pose)
    pyrender_scene.add_node(cam_node)

    color, depth = mesh_renderer.render(pyrender_scene)
    if need_seg:
        seg_dict = {node: int(node.name)+1 for i, node in enumerate(pyrender_scene.mesh_nodes)}
        seg = mesh_renderer.render(pyrender_scene, pyrender.RenderFlags.SEG, seg_dict)[0][..., 0:1]
    else:
        seg = None
    pyrender_scene.remove_node(cam_node)
    # del pyrender_scene
    return color, depth, seg

"""
For evaluation
"""

def iou_3d(obj1, obj2):
    '''the two bboxes are axis-aligned along Y, so the computation could be disentangled between X-Z plane and Y direction'''
    def get_corners(obj):
        center2corners = np.array([
            [1, 1, 1], [1, 1, -1], [-1, 1, -1], [-1, 1, 1], # Y upper level
            [1, -1, 1], [1, -1, -1], [-1, -1, -1], [-1, -1, 1] # Y lower level
        ]) * obj['size']
        corners = obj['translation'][:, np.newaxis] + obj['rotation'] @ center2corners.T
        return corners
    
    if obj1['obj'].label != obj2['obj'].label:
        return 0
    c1, c2 = get_corners(obj1), get_corners(obj2)
    if max(c1[1]) <= min(c2[1]) or max(c2[1]) <= min(c1[1]):
        return 0
    y_overlap = min(max(c1[1]), max(c2[1])) - max(min(c1[1]), min(c2[1]))
    p1 = Polygon([(c1[0, 0], c1[2, 0]), (c1[0, 1], c1[2, 1]), (c1[0, 2], c1[2, 2]), (c1[0, 3], c1[2, 3])])
    p2 = Polygon([(c2[0, 0], c2[2, 0]), (c2[0, 1], c2[2, 1]), (c2[0, 2], c2[2, 2]), (c2[0, 3], c2[2, 3])])
    xz_overlap = p1.intersection(p2).area
    if xz_overlap > 0:
        v1, v2 = np.prod(obj1['size']) * 8, np.prod(obj2['size']) * 8
        return (y_overlap * xz_overlap) / (v1 + v2 - y_overlap * xz_overlap)
    else:
        return 0

def get_blender_render(raw_source_scene, generate_trimesh, save_folder="./tmp_render/0", camera_dist=1.5,  verbose=True, num_images=8, remove_mesh=False):
    def get_floor(source_scene, renderables):
        all_vertices = np.concatenate([
            tr_mesh.vertices for tr_mesh in renderables
            ], axis=0)
        x_max, x_min = all_vertices[:, 0].max(), all_vertices[:, 0].min()
        z_max, z_min = all_vertices[:, 2].max(), all_vertices[:, 2].min()
        y_min = all_vertices[:, 1].min()
        trimesh_floor = source_scene.get_floor_plan(source_scene, rectangle_floor=True, room_size=[x_min, z_min, x_max, z_max])
        trimesh_floor.apply_translation([0, y_min, 0])
        return trimesh_floor
            
    mesh_save_folder = os.path.join(save_folder, "mesh")
    if os.path.exists(mesh_save_folder):
        shutil.rmtree(mesh_save_folder)
    os.makedirs(mesh_save_folder, exist_ok=True)
    generated_floor = get_floor(raw_source_scene, generate_trimesh)
    generate_trimesh.append(generated_floor)
    export_scene(mesh_save_folder, generate_trimesh)
    blender_render_scene(mesh_save_folder, save_folder, resolution_x=800, resolution_y=800, camera_dist=camera_dist, num_images=num_images, verbose=verbose)
    if remove_mesh:
        shutil.rmtree(mesh_save_folder)