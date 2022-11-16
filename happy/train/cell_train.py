import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR

from happy.train.utils import get_cell_confusion_matrix
from happy.models import resnet
from happy.data.setup_data import setup_cell_datasets
from happy.data.setup_dataloader import setup_dataloaders


def setup_data(
    organ,
    annotations_path,
    hp,
    image_size,
    num_workers,
    multiple_val_sets,
    val_batch,
    test_set=False,
):
    datasets = setup_cell_datasets(
        organ,
        annotations_path,
        hp.dataset_names,
        image_size,
        multiple_val_sets,
        test_set,
    )
    dataloaders = setup_dataloaders(False, datasets, num_workers, hp.batch, val_batch)
    return dataloaders


def setup_model(init_from_inc, out_features, pre_trained_path, frozen, device):
    if init_from_inc:
        model = resnet.build_resnet(out_features=out_features, depth=50)
    else:
        state_dict = torch.load(pre_trained_path, map_location=device)
        state_dict_num_outputs = state_dict["fc.output_layer.weight"].size()[0]
        model = resnet.build_resnet(out_features=out_features, depth=50)
        model.load_state_dict(state_dict, strict=True)

        if state_dict_num_outputs != out_features:
            num_features = model.fc[7].in_features
            model.fc[7] = nn.Linear(num_features, out_features)

    for child in model.children():
        for param in child.parameters():
            param.requires_grad = not frozen
    for param in model.fc.parameters():
        param.requires_grad = True

    # Move to GPU and define the optimiser
    model = model.to(device)
    print(f"Frozen layers is {frozen}")
    print("Model loaded to device")
    return model


def setup_training_params(
    model,
    learning_rate,
    train_dataloader,
    device,
    weighted_loss=True,
    decay_gamma=0.5,
    step_size=20,
):
    if weighted_loss:
        data = train_dataloader.dataset.all_annotations.class_name.map(
            train_dataloader.dataset.classes
        ).to_numpy()
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(data), y=data
        )
        class_weights = torch.FloatTensor(class_weights)
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        amsgrad=True,
    )
    scheduler = StepLR(optimizer, step_size=step_size, gamma=decay_gamma)
    return optimizer, criterion, scheduler


def train(
    organ,
    epochs,
    model,
    dataloaders,
    optimizer,
    criterion,
    logger,
    scheduler,
    run_path,
    device,
):
    prev_best_accuracy = 0
    batch_count = 0
    for epoch_num in range(epochs):
        model.train()
        # epoch recording metrics
        loss = {}
        predictions = {}
        ground_truth = {}

        for phase in dataloaders:
            print(phase)
            loss[phase] = []
            predictions[phase] = []
            ground_truth[phase] = []

            if phase != "train":
                model.eval()

            for i, data in enumerate(dataloaders[phase]):
                batch_loss, batch_preds, batch_truth, batch_count = single_batch(
                    phase,
                    optimizer,
                    criterion,
                    model,
                    data,
                    logger,
                    batch_count,
                    device,
                )
                # update epoch metrics
                loss[phase].append(batch_loss)
                predictions[phase].extend(batch_preds)
                ground_truth[phase].extend(batch_truth)
                if phase == "train":
                    logger.loss_hist.append(batch_loss)
                    print(
                        f"Epoch: {epoch_num} | Phase: {phase} | Iteration: {i} | "
                        f"Classification loss: {batch_loss:1.5f} | "
                        f"Running loss: {np.mean(logger.loss_hist):1.5f}"
                    )

            # Plot losses at each epoch for training and all validation sets
            log_epoch_metrics(logger, epoch_num, phase, loss, predictions, ground_truth)

        scheduler.step()

        # Calculate and plot confusion matrices for all validation sets
        print("Evaluating datasets")
        prev_best_accuracy = validate_model(
            organ,
            logger,
            epoch_num,
            prev_best_accuracy,
            model,
            run_path,
            predictions,
            ground_truth,
            list(dataloaders.keys()),
        )


def single_batch(phase, optimizer, criterion, model, data, logger, batch_count, device):
    optimizer.zero_grad()

    # Get predictions and calculate loss
    class_prediction = model(data["img"].to(device).float())
    loss = criterion(class_prediction, data["annot"].to(device))

    # Get predicted cell class and ground truth
    predictions = torch.max(class_prediction, 1)[1].cpu().tolist()
    ground_truths = data["annot"].tolist()

    # Plot training loss at each batch iteration
    if phase == "train":
        logger.log_batch_loss(batch_count, float(loss))
        batch_count += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

    return float(loss), predictions, ground_truths, batch_count


def log_epoch_metrics(logger, epoch_num, phase, loss, predictions, ground_truth):
    logger.log_loss(phase, epoch_num, np.mean(loss[phase]))
    accuracy = accuracy_score(ground_truth[phase], predictions[phase])
    logger.log_accuracy(phase, epoch_num, accuracy)


def validate_model(
    organ,
    logger,
    epoch_num,
    prev_best_accuracy,
    model,
    run_path,
    predictions,
    ground_truths,
    datasets,
):
    val_accuracy = logger.appenders["file"].train_stats.iloc[epoch_num][
        "val_all_accuracy"
    ]

    if val_accuracy > prev_best_accuracy:
        name = f"cell_model_accuracy_{round(val_accuracy, 4)}.pt"
        torch.save(model.state_dict(), run_path / name)
        print("Best model saved")

        # Generate confusion matrix for all the validation sets
        validation_confusion_matrices(
            organ,
            logger,
            predictions,
            ground_truths,
            datasets,
            run_path,
        )
        return val_accuracy
    else:
        return prev_best_accuracy


def validation_confusion_matrices(organ, logger, pred, truth, datasets, run_path):
    # Save confusion matrix plots for all validation sets
    datasets.remove("train")
    for dataset in datasets:
        cm = get_cell_confusion_matrix(organ, pred[dataset], truth[dataset])
        logger.log_confusion_matrix(cm, dataset, run_path)


def save_state(logger, model, hp, run_path):
    model.eval()
    torch.save(model.state_dict(), run_path / "cell_final_model.pt")
    hp.to_csv(run_path)
    logger.to_csv(run_path / "cell_train_stats.csv")
