import os
import pickle
from src.data.splits_builder import CSVSplitsBuilder
from src.data.threed_front import ThreedFront
from src.data.threed_front_dataset_edit import dataset_encoding_factory_edit

from constants import EDIT_DATA_FOLDER

def get_dataset(config, split=["train", "val"], is_eval=False, llm_plan_path=None):
    data_config = config["data"]

    if is_eval:
        edit_folder_path = os.path.join(EDIT_DATA_FOLDER, f"{data_config['filter_fn']}", "test_dataset")
    else:
        edit_folder_path = os.path.join(EDIT_DATA_FOLDER, f"{data_config['filter_fn']}", "train_dataset")
    if not os.path.exists(edit_folder_path):
        raise FileNotFoundError(f"Edit data folder {edit_folder_path} does not exist.")
    data_stats_path = os.path.join(EDIT_DATA_FOLDER, data_config['filter_fn'], "dataset_stats.txt")
    scene_data = ThreedFront.load_from_folder(edit_folder_path, path_to_train_stats=data_stats_path)
    print('loaded scene data', len(scene_data))

    splits_builder = CSVSplitsBuilder(data_config["annotation_file"])
    split_scene_ids = splits_builder.get_splits(split)
    new_scenes = []
    if isinstance(scene_data.scenes[0], str):
        for scene in scene_data.scenes:
            scene_id = scene.split("/")[-1].split(".")[0].split("_")[1]
            if scene_id in split_scene_ids:
                new_scenes.append(scene)
        uid_to_scene_index = {scene.split("/")[-1].split(".")[0]: i for i, scene in enumerate(new_scenes)}
    else:
        for scene in scene_data:
            if scene.scene_id in split_scene_ids:
                new_scenes.append(scene)
        uid_to_scene_index = {scene.uid: i for i, scene in enumerate(new_scenes)}

    data_stats_path = os.path.join(EDIT_DATA_FOLDER, data_config['filter_fn'], "dataset_stats.txt")
    new_scene_data = ThreedFront(new_scenes, path_to_train_stats=data_stats_path)
    new_scene_data.uid_to_scene_index = uid_to_scene_index

    encoding_type = data_config["encoding_type"]
    if is_eval:
        encoding_type += "_eval"
    processed_data = dataset_encoding_factory_edit(
        encoding_type,
        new_scene_data,
        data_config.get("augmentations", None),
        data_config.get("box_ordering", None),
        llm_plan_path=llm_plan_path,
    )

    print('loaded dataset')
    return new_scene_data, processed_data