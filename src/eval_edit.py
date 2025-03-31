
import sys, os
from pathlib import Path
import argparse
import json
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import numpy as np
import torch
import pickle
from src.data.utils_data import get_dataset
from lightning.pytorch import LightningModule, Trainer, seed_everything
from src.models.room_edit import RoomEdit

from constants import EDIT_DATA_FOLDER

def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument(
        "--sg_config_file",
        help="Path to the file that contains the scene graph generator configuration"
    )
    parser.add_argument(
        "--sg2sc_config_file",
        help="Path to the file that contains the scene graph to scene configuration"
    )
    parser.add_argument(
        "--output_directory",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--sg_weight_file",
        default=None,
        help=("The path to the scene graph generator weight file")
    )
    parser.add_argument(
        "--sg2sc_weight_file",
        default=None,
        help=("The path to the scene graph to scene weight file")
    )
    parser.add_argument(
        "--llm_plan_path",
        default=None,
        help=("The path to the llm plan file")
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=4,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the PRNG"
    )

    args = parser.parse_args(argv)
    seed_everything(args.seed)
    # Parse the config file
    with open(args.sg_config_file, "r") as f:
        sg_config = yaml.load(f, Loader=Loader)
    sg_config["load_path"] = args.sg_weight_file
    
    with open(args.sg2sc_config_file, "r") as f:
        sg2sc_config = yaml.load(f, Loader=Loader)
    sg2sc_config["load_path"] = args.sg2sc_weight_file
    
    test_data_tmp = f"{args.output_directory}/{sg_config['data']['filter_fn']}_test_eval_tmp.pkl"

    if not os.path.exists(test_data_tmp):
        raw_dataset, test_dataset = get_dataset(sg_config, split=["test"], is_eval=True)

        with open(test_data_tmp, 'wb') as f:
            pickle.dump([raw_dataset, test_dataset], f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading test data from tmp file")
        # train_dataset = torch.load(train_data_tmp)
        with open(test_data_tmp, 'rb') as f:
            raw_dataset, test_dataset = pickle.load(f)

    batch_size = 64
    num_devices = min(max(len(test_dataset)//batch_size, 1), torch.cuda.device_count())

    object_save_path = os.path.join(EDIT_DATA_FOLDER, sg_config['data']['filter_fn'], f"{sg_config['data']['filter_fn']}_objects.pkl")
    with open(object_save_path, "rb") as f:
        object_dataset = pickle.load(f)
    
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=args.n_processes,
            collate_fn=test_dataset.collate_fn,
            shuffle=False,
            pin_memory=True,
        )
    
    save_folder = f"{args.output_directory}/{sg_config['data']['dataset_type']}_template_plan_results"
    if args.llm_plan_path is not None:
        save_folder = save_folder.replace("template_plan", "llm_plan")
    model = RoomEdit(sg_config, sg2sc_config, 
                         num_objs=test_dataset.n_object_types, 
                         num_preds=test_dataset.n_predicate_types, 
                         pred_save_folder=save_folder, 
                         postprocess_func=test_dataset.post_process, 
                         object_dataset=object_dataset,
                         all_classes = test_dataset.class_labels,
                         raw_dataset=raw_dataset,)
    model.eval()

    trainer = Trainer(
                devices=num_devices,
                logger=None,
                default_root_dir=args.output_directory,
            )

    trainer.predict(model, test_dataloader, return_predictions=False)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Gather the results
    all_data = Path(save_folder).rglob("*scores.json")
    all_results = []
    for data_path in all_data:
        with open(data_path, "r") as f:
            data = json.load(f)
            all_results.append(data)

    new_dict = {}
    for r in all_results:
        for k, v in r.items():
            if k not in new_dict:
                new_dict[k] = []
            new_dict[k].append(v)
    
    for k, v in new_dict.items():
        new_dict[k] = np.array(v).mean()
    
    print(new_dict)
    with open(f"{save_folder}/full_results.json", "w") as f:
        json.dump(new_dict, f)
if __name__ == "__main__":
    main(sys.argv[1:])