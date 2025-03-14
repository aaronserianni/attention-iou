import os
import time

from bias import get_model, BCEWithLogitsLossFlipped
from waterbirds import WaterbirdsDataset, WaterbirdsTrainer
from celeb import CelebDataset, CelebTrainer
from coco import COCODataset, COCOTrainer
from load_args import get_trainer_args

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

def main():
    args = get_trainer_args()

    if args.waterbirds:
        print(f"Running bias training on Waterbirds with bias percentage {args.waterbirds_percentage} with model {args.model}")
        n_classes = 2
    elif args.celeb:
        print((f"Running bias training on CelebA with model target label {args.target_label} with model {args.model}" 
                f"{', one sided train ' + str(args.one_sided_train) + ' on ' + args.one_sided_target if args.one_sided_train is not None else ""}"
                f"{', target subgroup MCC ' + str(args.subgroup_train_mcc) + ' on ' + "-".join(args.subgroup_targets) + ' with range ' + str(args.subgroup_train_mcc_range) if args.subgroup_train_mcc is not None else ""}"))
        if len(args.target_label) == 1:
            args.target_label = args.target_label[0]
            n_classes = 40 if args.target_label == "all" else 2
        else:
            n_classes = len(args.target_label)     
    elif args.coco:
        print(f"Running bias training on COCO with model target label {args.target_label} with model {args.model}")
        if len(args.target_label) == 1:
            args.target_label = args.target_label[0]
            n_classes = 171 if args.target_label == "all" else 2
        else:
            n_classes = len(args.target_label)
    
    print("Loading data")
    if args.waterbirds:
        dataset = WaterbirdsDataset(
            os.path.join(os.path.abspath(args.data_dir), f"waterbird_complete{args.waterbirds_percentage}_forest2water2/"),
            args.resolution,
            augment_data=args.augment_data,
        )
    elif args.celeb:
        dataset = CelebDataset(
            os.path.abspath(args.data_dir),
            args.resolution,
            augment_data=args.augment_data,
            target_label=args.target_label,
            target_masks=args.target_masks,
            target_mcc=args.subgroup_train_mcc,
            target_mcc_range=args.subgroup_train_mcc_range,
            subgrouping_labels=args.subgroup_targets,
            one_sided_train=args.one_sided_train,
            one_sided_target=args.one_sided_target,
            flip_labels=args.flip_labels,
        )
    elif args.coco:
        dataset = COCODataset(
                os.path.abspath(args.data_dir),
                args.resolution,
                augment_data=args.augment_data,
                target_label=args.target_label,
                target_masks=args.target_masks,
            )
    print("Loaded data")

    (train_data, val_data, test_data) = dataset.prepare_dataloaders(
        train=True
    )

    train_loader = train_data.get_loader(
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = val_data.get_loader(
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = test_data.get_loader(
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    print("Created dataloaders")

    model = get_model(args.model, n_classes, args.weights)
    print("Number of trainable paramters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(os.path.join(args.tensorboard_dir, time.strftime("%c")))
    print("Tensorboard Dir: " + str(os.path.join(args.tensorboard_dir, time.strftime("%c"))))

    if args.model == "vit_b32":
        optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.000001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)

    if dataset.multi_target:
        loss_function = BCEWithLogitsLossFlipped(pos_weight=torch.from_numpy(dataset.loss_weights).to(device), flipped=args.flip_labels)
    else:
        loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(dataset.loss_weights).float().to(device))

    if args.waterbirds:
        trainer = WaterbirdsTrainer(
            model,
            dataset,
            device,
            optimizer,
            scheduler,
            loss_function,
            train_loader,
            val_loader,
            test_loader,
            writer,
            multiple_figures=args.multiple_figures,
            figure_frequency=args.figure_frequency
        )
    elif args.celeb:
        trainer = CelebTrainer(
            model,
            dataset,
            device,
            optimizer,
            scheduler,
            loss_function,
            train_loader,
            val_loader,
            test_loader if args.subgroup_train_mcc is None else val_loader,
            writer,
            args.target_label,
            args.target_masks,
            multiple_figures=args.multiple_figures,
            figure_frequency=args.figure_frequency
        )
    elif args.coco:
        trainer = COCOTrainer(
            model,
            dataset,
            device,
            optimizer,
            scheduler,
            loss_function,
            train_loader,
            val_loader,
            val_loader,
            writer,
            args.target_label,
            args.target_masks,
            multiple_figures=args.multiple_figures,
            figure_frequency=args.figure_frequency,
            label_dir=args.data_dir,
        )
    
    if args.train_multiple:
        for i in range(args.train_multiple):
            model = get_model(args.model, n_classes, args.weights).to(args.device)
            trainer.model = model

            trainer.train_model(num_epochs=args.num_epochs)
            trainer.test_model()

            save_path = trainer.get_model_path(trainer.multi_target, args, index=i)

            print(f"Saving model to {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_model, save_path))
    else:
        trainer.train_model(num_epochs=args.num_epochs)
        trainer.test_model()

        if args.save_model:
            save_path = trainer.get_model_path(trainer.multi_target, args, index=args.train_index)

            print(f"Saving model to {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_model, save_path))
            

if __name__ == '__main__':
    main()
