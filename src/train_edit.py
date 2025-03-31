
import sys, os

import argparse
import logging
import sys
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import numpy as np
import torch
import pickle
from src.data.utils_data import get_dataset
from src.models.sg_diffusion_vq_objfeat_edit import SgObjfeatVQDiffusionEdit
from src.models.sg2sc_diffusion_edit import Sg2ScDiffusionEdit
# torch.backends.cuda.matmul.allow_tf32 = True
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# torch.set_float32_matmul_precision("high")


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument(
        "--config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--output_directory",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=8,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_tag",
        default='default',
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
    )

    args = parser.parse_args(argv)
    seed_everything(args.seed)
    device_num = torch.cuda.device_count() if args.with_wandb_logger else 1
    # Parse the config file
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)

    os.makedirs(args.output_directory, exist_ok=True)

    train_data_tmp = f"{args.output_directory}/{config['data']['dataset_type']}_train_data.pkl"

    if not os.path.exists(train_data_tmp):
        _, train_dataset = get_dataset(config, ["train", "val"])

        with open(train_data_tmp, 'wb') as f:
            pickle.dump(train_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading train data from tmp file")
        # train_dataset = torch.load(train_data_tmp)
        with open(train_data_tmp, 'rb') as f:
            train_dataset = pickle.load(f)

    # _, train_dataset, config = get_dataset(config, config["training"].get("splits", ["train", "val"]))
    batch_size = max(64, config["training"]["batch_size"]//device_num) if args.with_wandb_logger else 64
    # batch_size = 2
    train_dataloder = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=args.n_processes,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True
    )

    val_data_tmp = f"{args.output_directory}/{config['data']['dataset_type']}_test_data.pkl"

    if not os.path.exists(val_data_tmp):
        _, val_dataset = get_dataset(config, ["test"], is_eval=True)
        with open(val_data_tmp, 'wb') as f:
            pickle.dump(val_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading val data from tmp file")
        # val_dataset = torch.load(val_data_tmp)
        with open(val_data_tmp, 'rb') as f:
            val_dataset = pickle.load(f)
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=args.n_processes,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    if "sg2sc" in config["model"]["name"]:
        model = Sg2ScDiffusionEdit(config, train_dataset.n_object_types, train_dataset.n_predicate_types)
    else:
        model = SgObjfeatVQDiffusionEdit(config, train_dataset.n_object_types, train_dataset.n_predicate_types)

    model_save_name = f"{config['data']['dataset_type']}_{config['model']['name']}"

    checkpoint_callback = ModelCheckpoint(
                dirpath=args.output_directory,
                filename=model_save_name + "-{epoch:02d}-{val_loss:.3f}",
                monitor="val_loss",
                save_top_k=2,
            )
    
    wandb_logger = WandbLogger(name=model_save_name, save_dir=args.output_directory, project="InstrucScene", offline=False, tags=[args.experiment_tag]) if args.with_wandb_logger else None
    trainer = Trainer(
        devices=device_num,
        max_epochs=config["training"]["epochs"],
        logger=wandb_logger,
        default_root_dir=args.output_directory,
        callbacks=[checkpoint_callback],
        strategy='ddp_find_unused_parameters_true' if "sg2sc" in config["model"]["name"] else 'ddp',
        log_every_n_steps=10
    )
    resume = None
    trainer.fit(model, train_dataloder, val_dataloader, ckpt_path=resume)

if __name__ == "__main__":
    main(sys.argv[1:])