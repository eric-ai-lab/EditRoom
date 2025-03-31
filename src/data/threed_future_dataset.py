# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import numpy as np
import pickle

import torch
from .utils import parse_threed_future_models
from tqdm import tqdm
from copy import deepcopy

class ThreedFutureDataset(object):
    def __init__(self, objects):
        assert len(objects) > 0
        self.objects = objects
        unique_jids = set()
        unique_objects = []
        unique_objects_dict = {}
        for oi in self.objects:
            if oi.model_jid not in unique_jids:
                unique_objects.append(oi)
                unique_jids.add(oi.model_jid)
                unique_objects_dict[oi.model_jid] = oi
        self._unique_objects_dict= unique_objects_dict
        self._unique_objects = unique_objects
        self._unique_object_feats = None
        self._all_object_feats = None

    def __len__(self):
        return len(self.objects)

    def __str__(self):
        return "Dataset contains {} objects with {} discrete types".format(
            len(self)
        )

    def __getitem__(self, idx):
        return self.objects[idx]

    def _filter_objects_by_label(self, label):
        if label is not None:
            return [oi for oi in self.objects if oi.label == label]
        else:  # return all objects if `label` is not specified
            return [oi for oi in self.objects]

    def get_closest_furniture_to_box(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = np.sum((oi.size - query_size)**2, axis=-1)
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x:x[1])]
        return sorted_mses[0]

    def get_closest_furniture_to_2dbox(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = (
                (oi.size[0] - query_size[0])**2 +
                (oi.size[2] - query_size[1])**2
            )
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x: x[1])]
        return sorted_mses[0]

    ################################ For InstructScene BEGIN ################################

    def get_closest_furniture_to_objfeat_and_size(self, query_label, query_size, query_objfeat, objfeat_type):
        # 1. Filter objects by label
        # 2. Sort objects by feature cosine similarity
        # 3. Pick top N objects (N=1 by default), i.e., select objects by feature cossim only
        # 4. Sort remaining objects by size MSE
        objects = self._filter_objects_by_label(query_label)

        cos_sims = {}
        for i, oi in enumerate(objects):
            query_objfeat = query_objfeat / np.linalg.norm(query_objfeat, axis=-1, keepdims=True)  # L2 normalize
            assert np.allclose(np.linalg.norm(eval(f"oi.{objfeat_type}_features"), axis=-1), 1.0)  # sanity check: already L2 normalized
            cos_sims[oi] = np.dot(eval(f"oi.{objfeat_type}_features"), query_objfeat)
        sorted_cos_sims = [k for k, v in sorted(cos_sims.items(), key=lambda x:x[1], reverse=True)]

        N = 1  # TODO: make it configurable
        filted_objects = sorted_cos_sims[:min(N, len(sorted_cos_sims))]
        mses = {}
        for i, oi in enumerate(filted_objects):
            mses[oi] = np.sum((oi.size - query_size)**2, axis=-1)
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x:x[1])]
        return sorted_mses[0], cos_sims[sorted_mses[0]]  # return values of cossim for debugging
    
    def get_closest_furniture_to_objfeat(self, query_label, query_objfeat, objfeat_type="openshape_vitg14"):
        # 1. Filter objects by label
        # 2. Sort objects by feature cosine similarity
        # 3. Pick top N objects (N=1 by default), i.e., select objects by feature cossim only
        # 4. Sort remaining objects by size MSE
        objects = self._filter_objects_by_label(query_label)
        query_objfeat = query_objfeat / np.linalg.norm(query_objfeat, axis=-1, keepdims=True)  # L2 normalize
        cos_sims = {}
        for i, oi in enumerate(objects):
            # assert np.allclose(np.linalg.norm(eval(f"oi.{objfeat_type}_features"), axis=-1), 1.0)  # sanity check: already L2 normalized
            cos_sims[oi] = np.dot(eval(f"oi.{objfeat_type}_features"), query_objfeat)
        sorted_cos_sims = [k for k, v in sorted(cos_sims.items(), key=lambda x:x[1], reverse=True)]

        N = 1  # TODO: make it configurable
        filted_objects = sorted_cos_sims[:min(N, len(sorted_cos_sims))]
        return deepcopy(filted_objects[0])

    ################################ For InstructScene END ################################

    @classmethod
    def from_dataset_directory(
        cls, dataset_directory, path_to_model_info, path_to_models
    ):
        objects = parse_threed_future_models(
            dataset_directory, path_to_models, path_to_model_info
        )
        return cls(objects)

    @classmethod
    def from_pickled_dataset(cls, path_to_pickled_dataset):
        with open(path_to_pickled_dataset, "rb") as f:
            dataset = pickle.load(f)
        return dataset
    
    @property
    def unique_objects(self):
        if not hasattr(self, "_unique_objects") or self._unique_objects is None:
            unique_jids = set()
            unique_objects = []
            for oi in self.objects:
                if oi.model_jid not in unique_jids:
                    unique_objects.append(oi)
                    unique_jids.add(oi.model_jid)
            self._unique_objects = unique_objects
        return self._unique_objects
    
    @property
    def unique_objects_dict(self):
        if not hasattr(self, "_unique_objects_dict") or self._unique_objects_dict is None:
            unique_objects_dict = {}
            for oi in self.objects:
                if oi.model_jid not in unique_objects_dict:
                    unique_objects_dict[oi.model_jid] = oi
            self._unique_objects_dict = unique_objects_dict
        return self._unique_objects_dict
    
    def unique_object_feats(self):
        if self._unique_object_feats is None:
            openshape_feats = []
            labels = []
            for oi in tqdm(self.unique_objects, desc="Computing unique object features"):
                openshape_feats.append(oi.openshape_vitg14_features)
                # radius.append(oi.get_radius())
                labels.append(oi.label)
            openshape_feats = np.stack(openshape_feats)
            labels = np.array(labels)

            self._unique_object_feats = {"openshape_vitg14": openshape_feats, "labels": labels}
        return self._unique_object_feats
    
    def get_furniture_by_jid(self, jid):
        try:
            return deepcopy(self.unique_objects_dict[jid])
        except KeyError:
            raise ValueError("Object with jid {} not found".format(jid))
    
    def get_closest_furniture_to_ulipfeats(self, query_ulip, query_shape=None, ulip_threshold=0.4, joint_distance=False, use_orig=False, query_radius=None, query_class=None, use_unique=False):
        if use_unique:
            all_object_feats = self.unique_object_feats()
            all_object = self.unique_objects
        else:
            all_object_feats = self.all_object_feats
            all_object = self.objects
        
        if query_class is not None:
            new_all_object_index = []
            for i, obj in enumerate(all_object):
                if obj.label == query_class:
                    new_all_object_index.append(i)
            if len(new_all_object_index) > 0:
                all_object = [all_object[i] for i in new_all_object_index]
                all_object_feats = {k: v[new_all_object_index] for k, v in all_object_feats.items()}
        # Normalize query ULIP feature
        query_ulip = query_ulip / np.linalg.norm(query_ulip, axis=-1)

        # Compute cosine similarity between query_ulip and all unique object ULIP features
        if use_orig:
            ulip_feats = all_object_feats['ulip_orig']
        else:
            ulip_feats = all_object_feats['ulip']
        cosine_similarities = np.dot(ulip_feats, query_ulip.T).flatten()

        sort_keys = []
        if query_radius is not None:
            all_radius = all_object_feats['radius']
            radius_diff = np.abs(all_radius - query_radius)
            radius_diff[radius_diff < 0.1] = 0
            sort_keys.insert(0, radius_diff)
        if query_shape is None:
            sort_keys.insert(0, -cosine_similarities)
        else:
            query_shape = query_shape / np.linalg.norm(query_shape, axis=-1)
            shape_feats = all_object_feats['shape']
            shape_distances = np.dot(shape_feats, query_shape.T).flatten()
            if joint_distance:
                shape_distances = (shape_distances + cosine_similarities)/2
            cosine_similarities[cosine_similarities > ulip_threshold] = 1
            sort_keys.insert(0,-cosine_similarities) 
            sort_keys.insert(0,-shape_distances)
        
        index = np.lexsort(sort_keys)[0]
        return deepcopy(all_object[index])


################################ For InstructScene BEGIN ################################

class ThreedFutureFeatureDataset(ThreedFutureDataset):
    def __init__(self, objects, objfeat_type: str):
        super().__init__(objects)

        self.objfeat_type = objfeat_type
        self.objfeat_dim = {
            "openshape_vitg14": 1280
        }[objfeat_type]

    def __getitem__(self, idx):
        obj = self.objects[idx]
        return {
            "jid": obj.model_jid,
            "objfeat": eval(f"obj.{self.objfeat_type}_features")
        }

    def collate_fn(self, samples):
        sample_batch = {
            "jids": [],     # str; (bs,)
            "objfeats": []  # Tensor; (bs, objfeat_dim)
        }

        for sample in samples:
            sample_batch["jids"].append(str(sample["jid"]))
            sample_batch["objfeats"].append(sample["objfeat"])
        sample_batch["objfeats"] = torch.from_numpy(np.stack(sample_batch["objfeats"], axis=0)).float()

        return sample_batch

################################ For InstructScene END ################################
