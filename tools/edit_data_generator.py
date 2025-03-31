import os
import json
from src.data import filter_function
from src.data.splits_builder import CSVSplitsBuilder
from src.data.threed_front import ThreedFront
from src.data.threed_future_dataset import ThreedFutureDataset
from tools.utils import preprocess_edits
import pickle
from lightning.pytorch import seed_everything
from copy import deepcopy
import argparse
import numpy as np
from tqdm import tqdm
from constants import *

def create_splits(scenes_dataset, test_scene_ids, pre_type_sample_num=None):
    def get_pair(scene, scene_dataset, uid_to_scene_index):
        if not hasattr(scene, "command"):
            return [False], [scene.uid]
        source_scene_index = uid_to_scene_index[scene.original_id]
        source_scene = scene_dataset[source_scene_index]
        has_command, source_id = get_pair(source_scene, scene_dataset, uid_to_scene_index)
        pair = [scene.uid]+source_id
        return [True]+has_command, pair
    
    def get_edit_type(command):
        if "add" in command:
            return "add"
        if "replace" in command:
            return "replace"
        if "remove" in command:
            return "remove"
        if "move" in command:
            return "move"
        if "rotate" in command:
            return "rotate"
        if "shrink" in command or "enlarge" in command:
            return "scale"
        else:
            raise ValueError(f"Invalid command: {command}")
        
    uid_to_scene_index = {scene.uid: i for i, scene in enumerate(scenes_dataset)}
    permuted_indices = np.random.permutation(len(scenes_dataset))
    train_scenes = []
    test_scenes = []
    used_scene_uids = []
    counters = {"add": 0, "replace": 0, "remove": 0, "move": 0, "rotate": 0, "scale": 0}
    for index in tqdm(permuted_indices):
        scene = scenes_dataset[index]
        if scene.scene_id in test_scene_ids:
            if scene.uid in used_scene_uids or not hasattr(scene, "command"):
                continue
            has_commands, all_pairs = get_pair(scene, scenes_dataset, uid_to_scene_index)
            revert_i = np.arange(len(has_commands))[::-1]
            for i in revert_i:
                has_command = has_commands[i]
                uid = all_pairs[i]
                if uid in used_scene_uids:
                    continue
                sub_scene_i = scenes_dataset[uid_to_scene_index[uid]]
                if has_command:
                    edit_type = get_edit_type(sub_scene_i.command)
                    if pre_type_sample_num is not None and counters[edit_type] >= pre_type_sample_num:
                        delattr(sub_scene_i, "command")
                    else:
                        counters[edit_type] += 1
                    test_scenes.append(sub_scene_i)
                    used_scene_uids.append(uid)
                else:
                    test_scenes.append(sub_scene_i)
                    used_scene_uids.append(uid)
        else:
            train_scenes.append(scene)
        all_counter_full = [v >= pre_type_sample_num for v in counters.values()] if pre_type_sample_num is not None else [False]*len(counters)
        if all(all_counter_full):
            break
    
    return train_scenes, test_scenes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Train a generative model on bounding boxes"
        )
    parser.add_argument(
        "--room_type",
        default="bedroom",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--num_max_pre_room",
        type=int,
        default=20,
        help="The number of rooms to be generated for each room type"
    )

    parser.add_argument(
        "--output_directory",
        type=str,
        default=EDIT_DATA_FOLDER,
        help="Path to the output directory"
    )

    parser.add_argument(
        "--pre_type_sample_num",
        type=int,
        default=None,
        help="Limit the number of samples for each edit type when creating the test set"
    )

    args = parser.parse_args()

    seed_everything(42)

    config = {
        "filter_fn":                 f"threed_front_{args.room_type}",
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": "configs/invalid_threed_front_rooms.txt",
        "path_to_invalid_bbox_jids": "configs/black_list.txt",
        "annotation_file":           f"configs/{args.room_type}_threed_front_splits.csv"
    }

    output_directory = args.output_directory
    output_directory = os.path.join(output_directory, f"{config['filter_fn']}")
    os.makedirs(output_directory, exist_ok=True)

    scenes_dataset = ThreedFront.from_dataset_directory(
        dataset_directory=PATH_TO_SCENE,
        path_to_models=PATH_TO_MODEL,
        path_to_model_info=os.path.join(PATH_TO_MODEL, "model_info.json"),
        filter_fn=filter_function(
            config, ["train", "val", "test"], False
        )
    )
    objects = {}
    for scene in scenes_dataset:
        for obj in scene.bboxes:
            objects[obj.model_jid] = deepcopy(obj)
    objects = [vi for vi in objects.values()]
    print("Loading dataset with {} objects".format(len(objects)))
    objects_dataset = ThreedFutureDataset(objects)
    object_save_path = os.path.join(output_directory, f"{config['filter_fn']}_objects.pkl")
    with open(object_save_path, "wb") as f:
        pickle.dump(objects_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    edited_scene_dataset = preprocess_edits(scenes_dataset, objects_dataset, num_max_pre_room = args.num_max_pre_room)

    print("Loading dataset with {} rooms".format(len(edited_scene_dataset)))

    tr_bounds = edited_scene_dataset.bounds["translations"]
    si_bounds = edited_scene_dataset.bounds["sizes"]
    an_bounds = edited_scene_dataset.bounds["angles"]

    dataset_stats = {
        "bounds_translations": tr_bounds[0].tolist() + tr_bounds[1].tolist(),
        "bounds_sizes": si_bounds[0].tolist() + si_bounds[1].tolist(),
        "bounds_angles": an_bounds[0].tolist() + an_bounds[1].tolist(),
        "class_labels": edited_scene_dataset.class_labels,
        "object_types": edited_scene_dataset.object_types,
        "class_frequencies": edited_scene_dataset.class_frequencies,
        "class_order": edited_scene_dataset.class_order,
        "count_furniture": edited_scene_dataset.count_furniture
    }

    if "openshape_vitg14_features" in edited_scene_dataset.bounds:
        feat_bounds = edited_scene_dataset.bounds["openshape_vitg14_features"]
        dataset_stats["bounds_openshape_vitg14_features"] = feat_bounds[0].tolist() + feat_bounds[1].tolist()

    data_stats_path = os.path.join(output_directory, "dataset_stats.txt")
    with open(data_stats_path, "w") as f:
        json.dump(dataset_stats, f)

    splits_builder = CSVSplitsBuilder(config["annotation_file"])
    test_scene_ids = splits_builder.get_splits(["test"])

    train_scenes, test_scenes = create_splits(edited_scene_dataset, test_scene_ids, pre_type_sample_num=args.pre_type_sample_num)

    train_scenes_dataset = ThreedFront(train_scenes, path_to_train_stats=data_stats_path)
    test_scenes_dataset = ThreedFront(test_scenes, path_to_train_stats=data_stats_path)

    train_scenes_save_path = os.path.join(output_directory, f"train_dataset")
    train_scenes_dataset.save_to_folder(train_scenes_save_path)

    test_scenes_save_path = os.path.join(output_directory, f"test_dataset")
    test_scenes_dataset.save_to_folder(test_scenes_save_path)
    print("Train scenes saved to {}".format(train_scenes_save_path))
    print("Test scenes saved to {}".format(test_scenes_save_path))

    


