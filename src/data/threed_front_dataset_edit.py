import random
import os
import json
from tqdm import tqdm
from .utils_text import reverse_rel
from copy import deepcopy
from .threed_front_dataset_base import *

current_folder = os.path.dirname(os.path.abspath(__file__))
full_object_descriptions = json.load(open(os.path.join(current_folder, "object_caption.json")))
full_object_descriptions_augment = json.load(open(os.path.join(current_folder, "object_caption_mistral7B.json")))
full_object_descriptions_augment2 = json.load(open(os.path.join(current_folder, "object_caption_blip_chatgpt.json")))

class Scale_CosinAngle_Edit(DatasetDecoratorBase):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for params in [sample_params['sources'], sample_params['targets']]:
            for k, v in params.items():
                if k == "angles":
                    # [cos, sin]
                    params[k] = np.concatenate([np.cos(v), np.sin(v)], axis=-1)
                
                elif k in bounds:
                    params[k] = Scale.scale(
                        v, bounds[k][0], bounds[k][1]
                    )
        return sample_params

    def post_process(self, s):
        bounds = self.bounds
        sample_params = {}
        for k, v in s.items():
            if k == "angles":
                # theta = arctan sin/cos y/x
                sample_params[k] = np.arctan2(v[:, 1:2], v[:, 0:1])
                
            elif k in bounds:
                sample_params[k] = Scale.descale(
                    v, bounds[k][0], bounds[k][1]
                )
            else:
                sample_params[k] = v
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 2

class Rotation_Edit(DatasetDecoratorBase):
    def __init__(self, dataset, min_rad=0.174533, max_rad=5.06145, fixed=False):
        super().__init__(dataset)
        self._min_rad = min_rad
        self._max_rad = max_rad
        self._fixed   = fixed
        
    @staticmethod
    def rotation_matrix_around_y(theta):
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        return R

    @property
    def rot_angle(self):
        if np.random.rand() < 0.5:
            return np.random.uniform(self._min_rad, self._max_rad)
        else:
            return 0.0
    
    @property
    def fixed_rot_angle(self):
        if np.random.rand() < 0.25:
            return np.pi * 1.5
        elif np.random.rand() < 0.50:
            return np.pi
        elif np.random.rand() < 0.75:
            return np.pi * 0.5
        else:
            return 0.0

    def __getitem__(self, idx):
        # Get the rotation matrix for the current scene
        if self._fixed:
            rot_angle = self.fixed_rot_angle
        else:
            rot_angle = self.rot_angle
        R = Rotation_Edit.rotation_matrix_around_y(rot_angle)

        sample_params = self._dataset[idx]
        instructions = sample_params["instructions"]

        if "***" in instructions:
            new_instructions = []
            directions = instructions.split("***")
            for d in directions:
                for key in ['front', 'behind', 'left', 'right', 'back']:
                    if key in d and len(d)<15:
                        try:
                            new_d = rotate_rel(d, rot_angle)
                            new_instructions.append(new_d)
                            break
                        except:
                            break
                else:
                    new_instructions.append(d)
            instructions = "***".join(new_instructions)
            sample_params["instructions"] = instructions
        for params in [sample_params['sources'], sample_params['targets']]:
            for k, v in params.items():
                if k == "translations":
                    params[k] = v.dot(R)
                elif k == "angles":
                    angle_min, angle_max = self.bounds["angles"]
                    params[k] = \
                        (v + rot_angle - angle_min) % (2 * np.pi) + angle_min
                elif k == "room_layout":
                    # Fix the ordering of the channels because it was previously
                    # changed
                    img = np.transpose(v, (1, 2, 0))
                    params[k] = np.transpose(rotate(
                        img, rot_angle * 180 / np.pi, reshape=False
                    ), (2, 0, 1))
                elif k == "relations":
                    relations = v.copy()
                    for i in range(relations.shape[0]):
                        s, p, o = relations[i]
                        new_p = self.predicate_types.index(
                            rotate_rel(self.predicate_types[p], rot_angle)
                        )
                        relations[i, 1] = new_p
                    params[k] = relations
        return sample_params

class JitterEdit(DatasetDecoratorBase):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        translate_jitter = np.random.normal(0, 0.01)
        size_jitter = np.random.normal(0, 0.01)
        for params in [sample_params['sources'], sample_params['targets']]:
            params["translations"] += translate_jitter
            params["sizes"] += size_jitter
            # for k, v in params.items():
                # if k in ["translations", "sizes", "angles"]:
                #     params[k] = v + np.random.normal(0, 0.01)
        return sample_params
    
class Add_Edit(DatasetDecoratorBase):
    def __init__(self, dataset, edit_pair, eval=False, edit_type=["add_remove", "pose_change", "object_replace"], permutation_keys=[], modified_samples=None): # 40
        super().__init__(dataset)
        self.eval = eval
        self.permutation_keys = permutation_keys
        if modified_samples is None:
            self.modified_samples = edit_pair

            print("Number of modified samples: ", len(self.modified_samples))
        else:
            self.modified_samples = modified_samples
            print("Number of modified samples Loaded from save file: ", len(self.modified_samples))
    
    def __len__(self):
        return len(self.modified_samples)
    
    def __getitem__(self, idx):
        # source, target, instruction = deepcopy(self.modified_samples[idx])
        source_index, target_index, instruction = self.modified_samples[idx]
        target = deepcopy(self._dataset[target_index])
        source = deepcopy(self._dataset[source_index])
        # if not self.eval:
        #     instruction = random.choice(instruction.split("\n"))
        # else:
        if type(instruction) == str:
            instruction = instruction.split("\n")[0]
            all_descriptions = []
            for instruc in instruction.split("[JID]"):
                if "[/JID]" in instruc:
                    obj_jid = instruc.split("[/JID]")[0]
                    target_object_description = full_object_descriptions[obj_jid]
                    if not self.eval:
                        target_object_description_aug = full_object_descriptions_augment[obj_jid]
                        target_object_description += target_object_description_aug

                        if obj_jid in full_object_descriptions_augment2:
                            target_object_description_aug2 = full_object_descriptions_augment2[obj_jid]
                            target_object_description += target_object_description_aug2

                    description = random.choice(target_object_description)
                    if description.endswith("."):
                        description = description[:-1]
                    all_descriptions.append(instruc.replace(f"{obj_jid}[/JID]", description))
                else:
                    all_descriptions.append(instruc)
            
            instruction = "".join(all_descriptions)
            instruction = instruction.strip()
            if not instruction.endswith("."):
                instruction += "."
        elif type(instruction) == list:
            new_instruction = []
            for _instruction in instruction:
                all_descriptions = []
                for instruc in _instruction.split("[JID]"):
                    if "[/JID]" in instruc:
                        obj_jid = instruc.split("[/JID]")[0]
                        target_object_description = full_object_descriptions[obj_jid]
                        if not self.eval:
                            target_object_description_aug = full_object_descriptions_augment[obj_jid]
                            target_object_description += target_object_description_aug

                            if obj_jid in full_object_descriptions_augment2:
                                target_object_description_aug2 = full_object_descriptions_augment2[obj_jid]
                                target_object_description += target_object_description_aug2

                        description = random.choice(target_object_description)
                        if description.endswith("."):
                            description = description[:-1]
                        all_descriptions.append(instruc.replace(f"{obj_jid}[/JID]", description))
                    else:
                        all_descriptions.append(instruc)
                
                _instruction = "".join(all_descriptions)
                _instruction = _instruction.strip()
                if not _instruction.endswith("."):
                    _instruction += "."
                new_instruction.append(_instruction)
            instruction = new_instruction

        if len(self.permutation_keys) and not self.eval:
            source_num_obj = source["class_labels"].shape[0]
            target_num_obj = target["class_labels"].shape[0]
            source_ordering = np.random.permutation(source_num_obj)
            if source_num_obj == target_num_obj:
                target_ordering = source_ordering
            else:
                assert abs(source_num_obj-target_num_obj)==1, "The number of objects should be the same or differ by 1"
                source_jids = source["jids"].copy()
                target_jids = target["jids"].copy()
                
                if source_num_obj > target_num_obj:
                    # remove situation
                    target_obj_index = -1
                    for i in range(target_num_obj):
                        if target_jids[i] != source_jids[i]:
                            target_obj_index = i
                            break
                    if target_obj_index == -1:
                        target_obj_index = target_num_obj
                    target_index = list(range(target_num_obj))
                    target_index.insert(target_obj_index, -1)
                    target_index = np.array(target_index)
                    target_ordering = target_index[source_ordering]
                    # remove the -1 in the target_ordering
                    target_ordering = target_ordering[target_ordering != -1]
                else:
                    # add situation
                    target_obj_index = -1
                    for i in range(source_num_obj):
                        if source_jids[i] != target_jids[i]:
                            target_obj_index = i
                            break
                    if target_obj_index == -1:
                        target_obj_index = source_num_obj
                    target_index = list(range(target_num_obj))
                    target_index.remove(target_obj_index)
                    target_index = np.array(target_index)
                    target_ordering = target_index[source_ordering]
                    # append target_obj_index to the end of target_ordering
                    target_ordering = np.append(target_ordering, target_obj_index)
                assert len(target_ordering) == target_num_obj, "The length of target_ordering should be the same as target_num_obj"

            for k in self.permutation_keys:
                if k in source:
                    source[k] = source[k][source_ordering]
                    target[k] = target[k][target_ordering]
            source_old_to_new = {old: new for new, old in enumerate(source_ordering)}
            target_old_to_new = {old: new for new, old in enumerate(target_ordering)}
            if "relations" in source:
                for i in range(len(source['relations'])):
                    s, p, o = source['relations'][i]
                    source['relations'][i] = (source_old_to_new[s], p, source_old_to_new[o])
                for i in range(len(target['relations'])):
                    s, p, o = target['relations'][i]
                    target['relations'][i] = (target_old_to_new[s], p, target_old_to_new[o])

        return {
            "sources": source,
            "targets": target,
            "instructions": instruction
        }

class SG2SC_Edit(DatasetDecoratorBase):
    def __init__(self, dataset, objfeat_type=None):
        super().__init__(dataset)
        self.objfeat_type = objfeat_type

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        max_length = self._dataset.max_length
        for params in [sample_params['sources'], sample_params['targets']]:
            params_new = {}
            for k, v in params.items():
                if k == "class_labels":
                    class_labels = np.copy(v)
                    class_ids = np.argmax(class_labels, axis=-1).astype(np.int64)
                    params_new["objs"] = class_ids

            params.update(params_new)
            # Add the number of bounding boxes in the scene
            params["length"] = params["class_labels"].shape[0]

            # padding to max_length
            objs = params["objs"]
            triples = params["relations"]
            boxes = np.concatenate([
                params["translations"],
                params["sizes"],
                params["angles"]
            ], axis=-1)
            objfeats = params['objfeat_vq_recon']
            objfeat_vq_indices = params["objfeat_vq_indices"]
            objfeat_vitg14_features = params.get("objfeat_vitg14_features", None)
            objs = np.pad(
                objs, (0, max_length - objs.shape[0]),
                mode="constant", constant_values=self.n_object_types
            )
            boxes = np.pad(
                boxes, ((0, max_length - boxes.shape[0]), (0, 0)),
                mode="constant", constant_values=0.
            )

            objfeats = np.pad(objfeats, ((0, max_length - objfeats.shape[0]), (0, 0)),
                        mode="constant", constant_values=0.
                    )

            objfeat_vitg14_features = np.pad(objfeat_vitg14_features, ((0, max_length - objfeat_vitg14_features.shape[0]), (0, 0)),
                        mode="constant", constant_values=0.
                    )
            
            edges = self.n_predicate_types * np.ones((max_length, max_length), dtype=np.int64)  # (n, n)
            for s, p, o in triples:
                rev_p = self.predicate_types.index(
                    reverse_rel(self.predicate_types[p])
                )
                edges[s, o] = p
                edges[o, s] = rev_p

            # objfeat_vq_indices_pad = np.random.randint(0, 64, size=(max_length, objfeat_vq_indices.shape[1]))  # TODO: make `64` configurable
            objfeat_vq_indices_pad = 64 * np.ones([max_length, objfeat_vq_indices.shape[1]]).astype(np.int64)
            objfeat_vq_indices_pad[:objfeat_vq_indices.shape[0]] = objfeat_vq_indices

            obj_mask = np.zeros(max_length, dtype=np.int64)  # (n,)
            obj_mask[:params["length"]] = 1

            params["objs"] = objs
            params["boxes"] = boxes
            params["edges"] = edges
            params["obj_masks"] = obj_mask
            params["objfeat_vq_indices"] = objfeat_vq_indices_pad
            params["objfeat_vq_recon"] = objfeats
            params['flatten_edges'] = edges[np.triu_indices(max_length, k=1)]
            params["objfeat_vitg14_features"] = objfeat_vitg14_features

        return sample_params

    def collate_fn(self, samples):
        batch_sources, batch_targets, batch_instructions = [], [], []
        for sample in samples:
            batch_sources.append(sample["sources"])
            batch_targets.append(sample["targets"])
            batch_instructions.append(sample["instructions"])
        
        gather_keys = ["objs", "boxes", "edges", "obj_masks", "objfeat_vq_indices", "objfeat_vq_recon", "flatten_edges", "objfeat_vitg14_features"]

        new_batch_sources = {}
        new_batch_targets = {}
        for s in batch_sources:
            for k, v in s.items():
                if k not in new_batch_sources:
                    if isinstance(v, dict):
                        new_batch_sources[k] = {}
                    else:
                        new_batch_sources[k] = []
                
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if kk not in new_batch_sources[k]:
                            new_batch_sources[k][kk] = []
                        new_batch_sources[k][kk].append(vv)
                else:
                    new_batch_sources[k].append(v)
        
        for t in batch_targets:
            for k, v in t.items():
                if k not in new_batch_targets:
                    if isinstance(v, dict):
                        new_batch_targets[k] = {}
                    else:
                        new_batch_targets[k] = []
                
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if kk not in new_batch_targets[k]:
                            new_batch_targets[k][kk] = []
                        new_batch_targets[k][kk].append(vv)
                else:
                    new_batch_targets[k].append(v)
        
        for k in gather_keys:
            source_v = torch.from_numpy(np.stack(new_batch_sources[k], axis=0))
            target_v = torch.from_numpy(np.stack(new_batch_targets[k], axis=0))
            if source_v.dtype == torch.float64:
                source_v = source_v.float()
            if target_v.dtype == torch.float64:
                target_v = target_v.float()
            new_batch_sources[k] = source_v
            new_batch_targets[k] = target_v

        new_samples = {
            "sources": new_batch_sources,
            "targets": new_batch_targets,
            "instructions": batch_instructions
        }
        return new_samples

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims

## Dataset encoding API
def dataset_encoding_factory_edit(
    name,
    dataset,
    augmentations=None,
    box_ordering=None,
    llm_plan_path = None,
) -> DatasetDecoratorBase:
    # NOTE: The ordering might change after augmentations so really it should
    #       be done after the augmentations. For class frequencies it is fine
    #       though.
    llm_plan = None
    if llm_plan_path:
        with open(llm_plan_path, "r") as f:
            llm_plan = json.load(f)

    uid_to_scene_index = dataset.uid_to_scene_index
    total_scenes = len(dataset)
    all_edit_pairs = []
    for i, scene in tqdm(enumerate(dataset), desc="Creating Index", total=total_scenes):
        if not hasattr(scene, "command"):
            continue
        if llm_plan is not None and scene.uid not in llm_plan:
            print(f"llm_plan is provied but Scene {scene.uid} not in llm_plan, skipping...")
            continue
        target_index = i
        source_index = uid_to_scene_index[scene.original_id]
        if llm_plan is not None:
            instruction = llm_plan[scene.uid]
        else:
            if isinstance(scene.command, list):
                instruction = []
                for c in scene.command:
                    instruction.append(c)
            else:
                instruction = scene.command
        all_edit_pairs.append((source_index, target_index, instruction))

    is_eval = "eval" in name
    if "cached" in name:
        dataset_collection = OrderedDataset(
            CachedDatasetCollection(dataset),
            ["class_labels", "translations", "sizes", "angles"],
            box_ordering=box_ordering
        )
    else:
        box_ordered_dataset = BoxOrderedDataset(
            dataset,
            box_ordering
        )
        # room_layout = RoomLayoutEncoder(box_ordered_dataset)
        class_labels = ClassLabelsEncoder(box_ordered_dataset)
        translations = TranslationEncoder(box_ordered_dataset)
        sizes = SizeEncoder(box_ordered_dataset)
        angles = AngleEncoder(box_ordered_dataset)
        jid_encoder = JIDEncoder(box_ordered_dataset)
        room_uid_encoder = RoomUidEncoder(box_ordered_dataset)
        relation_encoder = RoomRelationEncoder(box_ordered_dataset)
        vq_index_encoder = OpenShapeIndexEncoder(box_ordered_dataset)
        vq_recon_encoder = OpenShapeReconEncoder(box_ordered_dataset)
        open_shape_enocder = OpenShapeOrigEncoder(box_ordered_dataset)

        dataset_collection = DatasetCollection(
            # room_layout,
            class_labels,
            translations,
            sizes,
            angles,
            jid_encoder,
            room_uid_encoder,
            relation_encoder,
            vq_index_encoder,
            vq_recon_encoder,
            open_shape_enocder
        )

    if name == "basic":
        return DatasetCollection(
            class_labels,
            translations,
            sizes,
            angles
        )

    # Add scene graphs
    if "graph" in name or "desc" in name:
        print("Add [scene graphs] to the dataset")
        dataset_collection = Add_SceneGraph(dataset_collection)

    # Add scene descriptions
    if "desc" in name:
        if "seed" in name:
            seed = int(name.split("_")[-1])
        else:
            seed = None
        print("Add [scene descriptions] to the dataset")
        dataset_collection = Add_Description(dataset_collection, seed=seed)


    permute_keys = [
        "class_labels", "translations", "sizes", "angles", "descriptions",
        "jids", "objfeat_vq_indices", "objfeat_vq_recon", "objfeat_vitg14_features"
    ]

    dataset_collection = Add_Edit(dataset_collection, all_edit_pairs, eval=is_eval, permutation_keys=permute_keys)

    if isinstance(augmentations, list) and not is_eval and llm_plan is None:
        for aug_type in augmentations:
            if aug_type == "rotation":
                print("Apply [rotation] augmentation")
                dataset_collection = Rotation_Edit(dataset_collection)
            elif aug_type == "fixed_rotation":
                print("Applying [fixed rotation] augmentation")
                dataset_collection = Rotation_Edit(dataset_collection, fixed=True)
            elif aug_type == "jitter":
                print("Apply [jittering] augmentation")
                dataset_collection = JitterEdit(dataset_collection)

    # Scale the input
    print(f"Scale {list(dataset_collection.bounds.keys())}")
    if "sincos_angle" in name:
        print("Use [cos, sin] for angle encoding")
        dataset_collection = Scale_CosinAngle_Edit(dataset_collection)
    else:
        dataset_collection = Scale(dataset_collection)

    ################################################################

    if "sg2sc" in name or "sgdiffusion" in name:
        assert "graph" in name or "desc" in name, \
            "Add scene graphs to the dataset first (as conditions)."
        print("Use [Sg2Sc diffusion] model")
        return SG2SC_Edit(dataset_collection)
    else:
        raise NotImplementedError()
