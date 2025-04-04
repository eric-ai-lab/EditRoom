# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 
from typing import *
from collections import Counter
from dataclasses import dataclass
from functools import cached_property, reduce, lru_cache
import json
import os
import random
import torch
import shutil
import numpy as np
import pyrender
from PIL import Image

import trimesh
from trimesh import Trimesh
import subprocess

from .common import BaseScene

from constants import *

INDEX_FOLDER = os.path.join(PATH_TO_PREPROCESS, "openshape_vitg14_indexs")
RECON_FOLDER = os.path.join(PATH_TO_PREPROCESS, "openshape_vitg14_recon")
RELATION_FOLDER = os.path.join(PATH_TO_PREPROCESS, "room_relations")

current_folder = os.path.dirname(os.path.abspath(__file__))
full_object_descriptions = json.load(open(os.path.join(current_folder, "object_caption.json")))

def export_scene(
    output_dir: str,
    trimesh_meshes: List[Trimesh],
    bbox_meshes: Optional[List[Trimesh]]=None,
    names: Optional[List[str]]=None
) -> None:
    """Export the scene as a directory of `.obj`, `.mtl` and `.png` files."""
    if names is None:
        names = ["object_{:02d}.obj".format(i) for i in range(len(trimesh_meshes))]
        mtl_names = ["material_{:02d}".format(i) for i in range(len(trimesh_meshes))]
    else:
        mtl_names = ["material_"+n.replace(".obj", "") for n in names]

    if bbox_meshes is not None and len(bbox_meshes) > 0:
        for i, b in enumerate(bbox_meshes):
            b.export(os.path.join(output_dir, "bbox_{:02d}.obj".format(i)))

    for i, m in enumerate(trimesh_meshes):
        obj_out, tex_out = trimesh.exchange.obj.export_obj(m, return_texture=True)

        with open(os.path.join(output_dir, names[i]), "w") as f:
            f.write(
                obj_out.replace("material.mtl", mtl_names[i]+".mtl")\
                    .replace("material_0.mtl", mtl_names[i]+".mtl")
            )

        # No material and texture to rename
        if tex_out is None:
            continue

        mtl_key = next(k for k in tex_out.keys() if k.endswith(".mtl"))
        path_to_mtl_file = os.path.join(output_dir, mtl_names[i]+".mtl")
        with open(path_to_mtl_file, "wb") as f:
            f.write(
                tex_out[mtl_key].replace(b"material_0.png", (mtl_names[i]+".png").encode("ascii"))\
                    .replace(b"material_0.jpeg", (mtl_names[i]+".jpeg").encode("ascii"))
            )
        tex_key = next(k for k in tex_out.keys() if not k.endswith(".mtl"))
        tex_ext = os.path.splitext(tex_key)[1]
        path_to_tex_file = os.path.join(output_dir, mtl_names[i]+tex_ext)
        with open(path_to_tex_file, "wb") as f:
            f.write(tex_out[tex_key])

def _blender_binary_path() -> str:
    path = os.getenv("BLENDER_PATH", None)
    if path is not None:
        return path

    if os.path.exists("blender/blender-3.3.1-linux-x64/blender"):
        return "blender/blender-3.3.1-linux-x64/blender"

    raise EnvironmentError(
        "To render 3D models, install Blender version 3.3.1 or higher and "
        "set the environment variable `BLENDER_PATH` to the path of the Blender executable."
    )

def blender_render_scene(
    scene_dir: str,
    output_dir: str,
    output_suffix="",
    *,
    engine="CYCLES",
    top_down_view=False,
    num_images=8,
    camera_dist=1.2,
    resolution_x=1024,
    resolution_y=1024,
    cycle_samples=32,
    verbose=False,
    timeout=15*60.,
):
    BLENDER_SCRIPT_PATH = "src/utils/blender_script.py"

    args = [
        _blender_binary_path(),
        "-b", "-P", BLENDER_SCRIPT_PATH,
        "--",
        "--scene_dir", scene_dir,
        "--output_dir", output_dir,
        "--output_suffix", output_suffix,
        "--engine", engine,
        "--num_images", str(num_images),
        "--camera_dist", str(camera_dist),
        "--resolution_x", str(resolution_x),
        "--resolution_y", str(resolution_y),
        "--cycle_samples", str(cycle_samples),
    ]
    if top_down_view:
        args += ["--top_down_view"]

    # Execute the command
    if verbose:
        subprocess.check_call(args)
    else:
        try:
            _ = subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=timeout)  # return stdout
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"{exc}: {exc.output}") from exc
        
def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))
    
def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)

def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T

def rotation_matrix(axis, theta):
    """Axis-angle rotation matrix from 3D-Front-Toolbox."""
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


@dataclass
class Asset:
    """Contains the information for each 3D-FUTURE model."""
    super_category: str
    category: str
    style: str
    theme: str
    material: str

    @property
    def label(self):
        return self.category


class ModelInfo(object):
    """Contains all the information for all 3D-FUTURE models.

        Arguments
        ---------
        model_info_data: list of dictionaries containing the information
                         regarding the 3D-FUTURE models.
    """
    def __init__(self, model_info_data):
        self.model_info_data = model_info_data
        self._model_info = None
        # List to keep track of the different styles, themes
        self._styles = []
        self._themes = []
        self._categories = []
        self._super_categories = []
        self._materials = []

    @property
    def model_info(self):
        if self._model_info is None:
            self._model_info = {}
            # Create a dictionary of all models/assets in the dataset
            for m in self.model_info_data:
                # Keep track of the different styles
                if m["style"] not in self._styles and m["style"] is not None:
                    self._styles.append(m["style"])
                # Keep track of the different themes
                if m["theme"] not in self._themes and m["theme"] is not None:
                    self._themes.append(m["theme"])
                # Keep track of the different super-categories
                if m["super-category"] not in self._super_categories and m["super-category"] is not None:
                    self._super_categories.append(m["super-category"])
                # Keep track of the different categories
                if m["category"] not in self._categories and m["category"] is not None:
                    self._categories.append(m["category"])
                # Keep track of the different categories
                if m["material"] not in self._materials and m["material"] is not None:
                    self._materials.append(m["material"])

                super_cat = "unknown_super-category"
                cat = "unknown_category"

                if m["super-category"] is not None:
                    super_cat = m["super-category"].lower().replace(" / ", "/")

                if m["category"] is not None:
                    cat = m["category"].lower().replace(" / ", "/")

                self._model_info[m["model_id"]] = Asset(
                    super_cat,
                    cat, 
                    m["style"],
                    m["theme"],
                    m["material"]
                )

        return self._model_info

    @property
    def styles(self):
        return self._styles

    @property
    def themes(self):
        return self._themes

    @property
    def materials(self):
        return self._materials

    @property
    def categories(self):
        return set([s.lower().replace(" / ", "/") for s in self._categories])

    @property
    def super_categories(self):
        return set([
            s.lower().replace(" / ", "/")
            for s in self._super_categories
        ])

    @classmethod
    def from_file(cls, path_to_model_info):
        with open(path_to_model_info, "rb") as f:
            model_info = json.load(f)

        return cls(model_info)


class BaseThreedFutureModel(object):
    def __init__(self, model_uid, model_jid, position, rotation, scale):
        self.model_uid = model_uid
        self.model_jid = model_jid
        self.position = position
        self.rotation = rotation
        self.scale = scale

    def _transform(self, vertices):
        # the following code is adapted and slightly simplified from the
        # 3D-Front toolbox (json2obj.py). It basically scales, rotates and
        # translates the model based on the model info.
        ref = [0, 0, 1]
        axis = np.cross(ref, self.rotation[1:])
        theta = np.arccos(np.dot(ref, self.rotation[1:]))*2
        vertices = vertices * self.scale
        if np.sum(axis) != 0 and not np.isnan(theta):
            R = rotation_matrix(axis, theta)
            vertices = vertices.dot(R.T)
        vertices += self.position

        return vertices


class ThreedFutureModel(BaseThreedFutureModel):
    def __init__(
        self,
        model_uid,
        model_jid,
        model_info,
        position,
        rotation,
        scale,
        path_to_models
    ):
        super().__init__(model_uid, model_jid, position, rotation, scale)
        self.model_info = model_info
        self.path_to_models = path_to_models
        self._label = None

    @property
    def raw_model_path(self):
        return os.path.join(
            PATH_TO_MODEL,
            self.model_jid,
            "raw_model.obj"
        )

    @property
    def texture_image_path(self):
        return os.path.join(
            PATH_TO_MODEL,
            self.model_jid,
            "texture.png"
        )

    @property
    def path_to_bbox_vertices(self):
        return os.path.join(
            PATH_TO_MODEL,
            self.model_jid,
            "bbox_vertices.npy"
        )

    @property
    def path_to_openshape_vitg14_index(self):
        return os.path.join(INDEX_FOLDER, f"{self.model_jid}.npy")
    
    @property
    def path_to_openshape_vitg14_recon(self):
        return os.path.join(RECON_FOLDER, f"{self.model_jid}.npy")
    
    ################################ For InstructScene BEGIN ################################

    @property
    def path_to_openshape_vitg14_features(self):
        return os.path.join(
            PATH_TO_MODEL,
            self.model_jid,
            "openshape_pointbert_vitg14.npy"
        )

    ################################ For InstructScene END ################################

    def raw_model(self):
        try:
            return trimesh.load(
                self.raw_model_path,
                process=False,
                force="mesh",
                skip_materials=True,
                skip_texture=True
            )
        except:
            import pdb
            pdb.set_trace()
            print("Loading model failed", flush=True)
            print(self.raw_model_path, flush=True)
            raise

    def raw_model_transformed(self, offset=[[0, 0, 0]]):
        model = self.raw_model()
        faces = np.array(model.faces)
        vertices = self._transform(np.array(model.vertices)) + offset

        return trimesh.Trimesh(vertices, faces)

    ################################ For InstructScene BEGIN ################################

    @cached_property
    def openshape_vitg14_features(self):
        if os.path.exists(self.path_to_openshape_vitg14_features):
            latent = np.load(self.path_to_openshape_vitg14_features).astype(np.float32)
            return latent
        else:
            return None

    ################################ For InstructScene END ################################

    def openshape_vitg14_index(self):
        index = np.load(self.path_to_openshape_vitg14_index)
        return index
    
    def openshape_vitg14_recon(self):
        recon = np.load(self.path_to_openshape_vitg14_recon).astype(np.float32)
        return recon

    def centroid(self, offset=[[0, 0, 0]]):
        return self.corners(offset).mean(axis=0)

    @cached_property
    def size(self):
        corners = self.corners()
        return np.array([
            np.sqrt(np.sum((corners[4]-corners[0])**2))/2,
            np.sqrt(np.sum((corners[2]-corners[0])**2))/2,
            np.sqrt(np.sum((corners[1]-corners[0])**2))/2
        ])

    def bottom_center(self, offset=[[0, 0, 0]]):
        centroid = self.centroid(offset)
        size = self.size
        return np.array([centroid[0], centroid[1]-size[1], centroid[2]])

    @cached_property
    def bottom_size(self):
        return self.size * [1, 2, 1]

    @cached_property
    def z_angle(self):
        # See BaseThreedFutureModel._transform for the origin of the following
        # code.
        ref = [0, 0, 1]
        axis = np.cross(ref, self.rotation[1:])
        theta = np.arccos(np.dot(ref, self.rotation[1:]))*2

        if np.sum(axis) == 0 or np.isnan(theta):
            return 0

        assert np.dot(axis, [1, 0, 1]) == 0
        assert 0 <= theta <= 2*np.pi

        if theta >= np.pi:
            theta = theta - 2*np.pi

        return np.sign(axis[1]) * theta

    @property
    def label(self):
        if self._label is None:
            self._label = self.model_info.label
        return self._label

    @label.setter
    def label(self, _label):
        self._label = _label

    def corners(self, offset=[[0, 0, 0]]):
        try:
            bbox_vertices = np.load(self.path_to_bbox_vertices, mmap_mode="r")
        except:
            bbox_vertices = np.array(self.raw_model().bounding_box.vertices)
            np.save(self.path_to_bbox_vertices, bbox_vertices)
        c = self._transform(bbox_vertices)
        return c + offset

    def one_hot_label(self, all_labels):
        return np.eye(len(all_labels))[self.int_label(all_labels)]

    def int_label(self, all_labels):
        return all_labels.index(self.label)

    def copy_from_other_model(self, other_model):
        model = ThreedFutureModel(
            model_uid=other_model.model_uid,
            model_jid=other_model.model_jid,
            model_info=other_model.model_info,
            position=self.position,
            rotation=self.rotation,
            scale=other_model.scale,
            path_to_models=self.path_to_models
        )
        model.label = self.label
        return model
    
    def description(self, need_all=False):
        descriptions = full_object_descriptions[self.model_jid]
        if need_all and len(descriptions) > 1:
            description = descriptions
        elif len(descriptions) >= 1:
            description = random.choice(descriptions)
        else:
            description = self.label
        return description

    def trimesh_transformed(self, offset=[0, 0, 0]):
        model = trimesh.load(
                self.raw_model_path,
                force="mesh",
            )
        model.visual.material.image = Image.open(self.texture_image_path)
        model.apply_scale(self.scale)

        ref = [0, 0, 1]
        axis = np.cross(ref, self.rotation[1:])
        theta = np.arccos(np.dot(ref, self.rotation[1:]))*2
        if np.sum(axis) != 0 and not np.isnan(theta):
            R = rotation_matrix(axis, theta)
            transform = np.eye(4)
            transform[:3, :3] = R

            model.apply_transform(transform)
        model.apply_translation(np.array(self.position)+np.array(offset))

        return model


class ThreedFutureExtra(BaseThreedFutureModel):
    def __init__(
        self,
        model_uid,
        model_jid,
        xyz,
        faces,
        model_type,
        position,
        rotation,
        scale
    ):
        super().__init__(model_uid, model_jid, position, rotation, scale)
        self.xyz = xyz
        self.faces = faces
        self.model_type = model_type

    def raw_model_transformed(self, offset=[[0, 0, 0]]):
        vertices = self._transform(np.array(self.xyz)) + offset
        faces = np.array(self.faces)
        return trimesh.Trimesh(vertices, faces)


class Room(BaseScene):
    def __init__(
        self, scene_id, scene_type, bboxes, extras, json_path,
        path_to_room_masks_dir=None
    ):
        super().__init__(scene_id, scene_type, bboxes)
        self.json_path = json_path
        self.extras = extras

        self.uid = "_".join([self.json_path, scene_id])
        self.path_to_room_masks_dir = path_to_room_masks_dir
        if path_to_room_masks_dir is not None:
            self.path_to_room_mask = os.path.join(
                self.path_to_room_masks_dir, self.uid, "room_mask.png"
            )
        else:
            self.path_to_room_mask = None

    @property
    def floor(self):
        return [ei for ei in self.extras if ei.model_type == "Floor"][0]

    @property
    @lru_cache(maxsize=512)
    def bbox(self):
        corners = np.empty((0, 3))
        for f in self.bboxes:
            corners = np.vstack([corners, f.corners()])
        return np.min(corners, axis=0), np.max(corners, axis=0)

    @cached_property
    def bboxes_centroid(self):
        a, b = self.bbox
        return (a+b)/2

    @property
    def furniture_in_room(self):
        return [f.label for f in self.bboxes]

    @property
    def floor_plan(self):
        def cat_mesh(m1, m2):
            v1, f1 = m1
            v2, f2 = m2
            v = np.vstack([v1, v2])
            f = np.vstack([f1, f2 + len(v1)])
            return v, f

        # Compute the full floor plan
        vertices, faces = reduce(
            cat_mesh,
            ((ei.xyz, ei.faces) for ei in self.extras if ei.model_type == "Floor")
        )
        return np.copy(vertices), np.copy(faces)

    @cached_property
    def floor_plan_bbox(self):
        vertices, faces = self.floor_plan
        return np.min(vertices, axis=0), np.max(vertices, axis=0)

    @cached_property
    def floor_plan_centroid(self):
        a, b = self.floor_plan_bbox
        return (a+b)/2

    @cached_property
    def centroid(self):
        return self.floor_plan_centroid

    @property
    def count_furniture_in_room(self):
        return Counter(self.furniture_in_room)

    @property
    def room_mask(self):
        return self.room_mask_rotated(0)
    
    @property
    def relation_path(self):
        return os.path.join(RELATION_FOLDER, f"{self.uid}.npy")

    def room_mask_rotated(self, angle=0):
        # The angle is in rad
        im = Image.open(self.path_to_room_mask).convert("RGB")
        # Downsample the room_mask image by applying bilinear interpolation
        im = im.rotate(angle * 180 / np.pi, resample=Image.BICUBIC)

        return np.asarray(im).astype(np.float32) / np.float32(255)

    def category_counts(self, class_labels):
        """List of category counts in the room
        """
        print(class_labels)
        if "start" in class_labels and "end" in class_labels:
            class_labels = class_labels[:-2]
        category_counts = [0]*len(class_labels)

        for di in self.furniture_in_room:
            category_counts[class_labels.index(di)] += 1
        return category_counts

    def ordered_bboxes_with_centroid(self):
        centroids = np.array([f.centroid(-self.centroid) for f in self.bboxes])
        ordering = np.lexsort(centroids.T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes

    def ordered_bboxes_with_class_labels(self, all_labels):
        centroids = np.array([f.centroid(-self.centroid) for f in self.bboxes])
        int_labels = np.array(
            [[f.int_label(all_labels)] for f in self.bboxes]
        )
        ordering = np.lexsort(np.hstack([centroids, int_labels]).T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes

    def ordered_bboxes_with_class_frequencies(self, class_order):
        centroids = np.array([f.centroid(-self.centroid) for f in self.bboxes])
        label_order = np.array([
            [class_order[f.label]] for f in self.bboxes
        ])
        ordering = np.lexsort(np.hstack([centroids, label_order]).T)
        ordered_bboxes = [self.bboxes[i] for i in ordering[::-1]]

        return ordered_bboxes

    def augment_room(self, objects_dataset):
        bboxes = self.bboxes
        # Randomly pick an asset to be augmented
        bi = np.random.choice(self.bboxes)
        query_label = bi.label
        query_size = bi.size + np.random.normal(0, 0.02)
        # Retrieve the new asset based on the size of the picked asset
        furniture = objects_dataset.get_closest_furniture_to_box(
            query_label, query_size
        )
        bi_retrieved = bi.copy_from_other_model(furniture)

        new_bboxes = [
            box for box in bboxes if not box == bi
        ] + [bi_retrieved]

        return Room(
            scene_id=self.scene_id + "_augm",
            scene_type=self.scene_type,
            bboxes=new_bboxes,
            extras=self.extras,
            json_path=self.json_path,
            path_to_room_masks_dir=self.path_to_room_masks_dir
        )
    
    def get_room_description(self):
        total_descriptions = []
        for i, obj in enumerate(self.bboxes):
            label = obj.label.replace("_", " ")
            size = [round(s, 2) for s in obj.size.tolist()]
            z_angle = round(np.rad2deg(obj.z_angle))
            position = [round(p, 2) for p in obj.position]
            obj_description = {
                "class": label,
                "size": size,
                "vertical angle": z_angle,
                "centroid": position,
                "description": obj.description()
            }
            obj_description = json.dumps(obj_description)
            total_descriptions.append(f"Object {i}: {obj_description}")
        scene_description = " \n ".join(total_descriptions)
        return scene_description

    def get_renderable_objects(self, with_objects_offset=True, with_floor_plan_offset=False, with_floor_plan=False):
        if with_objects_offset:
            offset = -self.bboxes_centroid
        elif with_floor_plan_offset:
            offset = -self.floor_plan_centroid
        else:
            offset = [0, 0, 0]

        renderables = [ # deal with multiple same jid objects
            f.trimesh_transformed(offset=offset) for f in self.bboxes
        ]
        if with_floor_plan:
            all_vertices = np.concatenate([
                tr_mesh.vertices for tr_mesh in renderables
            ], axis=0)
            x_max, x_min = all_vertices[:, 0].max(), all_vertices[:, 0].min()
            z_max, z_min = all_vertices[:, 2].max(), all_vertices[:, 2].min()
            y_min = all_vertices[:, 1].min()
            trimesh_floor = self.get_floor_plan(self, rectangle_floor=True, room_size=[x_min, z_min, x_max, z_max])
            trimesh_floor.apply_translation([0, y_min, 0])
            renderables.append(trimesh_floor)
        # scene = trimesh.Scene(renderables)
        return renderables

    def render(self,
               renderables,
               elevation=0,
                azimuth=0,
                radius=None,
                resolution=512,
                orthogonal_view=False):
        
        trimesh_scene = trimesh.Scene(renderables)
        # combine all renderables (trimesh meshes) into a trimesh scene
        if radius is None:
            radius = (np.sum(trimesh_scene.bounding_box.extents**2)**0.5 / 2)+2
        target = trimesh_scene.centroid.copy()

        cam_pose = orbit_camera(elevation, azimuth, radius, target=target)

        scene = pyrender.Scene.from_trimesh_scene(trimesh_scene, ambient_light=(1.0, 1.0, 1.0), bg_color=(0.5, 0.5, 0.5, 1.0))
    
        mesh_renderer = pyrender.OffscreenRenderer(resolution, resolution)
        if orthogonal_view:
            camera = pyrender.OrthographicCamera(xmag=2.0, ymag=2.0, znear=0.1, zfar=100)
        else:
            camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(49.1), aspectRatio=1.0, znear=0.1, zfar=100)
        cam_node = pyrender.Node(camera=camera, matrix=cam_pose)
        scene.add_node(cam_node)

        color, depth = mesh_renderer.render(scene)
        mesh_renderer.delete()
        return color
    
    def direct_render(self, save_path,
                      elevation=0,
                azimuth=0,
                radius=None,
                resolution=1024,
                orthogonal_view=False):
        renderables = self.get_renderable_objects(with_floor_plan=False)
        img = self.render(renderables, elevation, azimuth, radius, resolution, orthogonal_view)
        return Image.fromarray(img).save(save_path)
    
    def get_blender_render(self, save_folder, camera_dist=1.5, num_images=8, top_down_view=False, verbose=True):
        mesh_save_folder = os.path.join(save_folder, "mesh")
        if os.path.exists(mesh_save_folder):
            shutil.rmtree(mesh_save_folder)
        os.makedirs(mesh_save_folder, exist_ok=True)
        generate_trimesh =self.get_renderable_objects(with_floor_plan=True)
        export_scene(mesh_save_folder, generate_trimesh)
        blender_render_scene(mesh_save_folder, save_folder, resolution_x=800, resolution_y=800, camera_dist=camera_dist, num_images=num_images, top_down_view=top_down_view, verbose=verbose)
    
    def get_llm_command(self, save_folder):
        path = os.path.join(save_folder, f"{self.uid}.json")
        with open(path, "r") as f:
            command = json.load(f)
        return command
    
    @staticmethod
    def get_floor_plan(
        scene,
        floor_textures: List[str]=["configs/floor_plan_texture_images/floor_00003.jpg"],
        rectangle_floor=True,
        room_size: Optional[Union[torch.Tensor, np.ndarray, List[float]]]=None,
        room_angle: Optional[float]=None
    ):
        """Get a trimesh object of the floor plan with a random texture."""
        vertices, faces = scene.floor_plan
        vertices = vertices - scene.floor_plan_centroid
        uv = np.copy(vertices[:, [0, 2]])
        uv -= uv.min(axis=0)
        uv /= 0.3  # repeat every 30cm
        texture = np.random.choice(floor_textures)

        if rectangle_floor:
            floor_sizes = room_size if room_size is not None else \
                ((np.max(vertices, axis=0) - np.min(vertices, axis=0)) / 2.)
            if len(floor_sizes) == 3:
                floor_4corners = np.array([
                    [-floor_sizes[0], 0., -floor_sizes[2]],
                    [-floor_sizes[0], 0.,  floor_sizes[2]],
                    [ floor_sizes[0], 0.,  floor_sizes[2]],
                    [ floor_sizes[0], 0., -floor_sizes[2]],
                ])
            elif len(floor_sizes) == 4:
                floor_4corners = np.array([
                    [floor_sizes[0], 0., floor_sizes[1]],  # left bottom
                    [floor_sizes[0], 0., floor_sizes[3]],  # left top
                    [floor_sizes[2], 0., floor_sizes[3]],  # right top
                    [floor_sizes[2], 0., floor_sizes[1]],  # right bottom
                ])
            else:
                raise ValueError(f"Invalid floor size: {floor_sizes}")
            vertices = floor_4corners
            faces = np.array([[0, 1, 2], [0, 2, 3]])

        if room_angle is not None:
            R = np.zeros((3, 3))
            R[0, 0] = np.cos(room_angle)
            R[0, 2] = -np.sin(room_angle)
            R[2, 0] = np.sin(room_angle)
            R[2, 2] = np.cos(room_angle)
            R[1, 1] = 1.
            vertices = vertices.dot(R)

        tr_floor = trimesh.Trimesh(np.copy(vertices), np.copy(faces), process=False)
        tr_floor.visual = trimesh.visual.TextureVisuals(
            uv=np.copy(uv),
            material=trimesh.visual.material.SimpleMaterial(image=Image.open(texture))
        )

        return tr_floor