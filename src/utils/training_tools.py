import torch
import sys
import pandas as pd
import os
from config import __dir__, __parent__dir__
from .utils import timer
from .logger import Logger

__testing__ = False


def save_losses(epoch, training_losses, val_losses, filename, epoch_range=5):
    filename_of_logs = f"{filename}_train_val_logs.csv"

    epoch_ranged = list(range(epoch - epoch_range + 1, epoch + 1))

    n_epoch = len(epoch_ranged)
    n_train = len(training_losses)
    n_val = len(val_losses)

    # balance training loss length
    if n_train != n_epoch:
        diff = n_train - n_epoch
        Logger.warning("training_losses length mismatch; adjusting...")
        if diff < 0:
            # extend training losses
            training_losses.extend([0] * abs(diff))
        else:
            # extend epoch list
            epoch_ranged.extend(range(epoch + 1, epoch + diff + 1))

    # balance val loss length  (NEW)
    if n_val != len(epoch_ranged):
        diff = n_val - len(epoch_ranged)
        Logger.warning("val_losses length mismatch; adjusting...")
        if diff < 0:
            # extend val losses
            val_losses.extend([0] * abs(diff))
        else:
            # extend epoch list
            last_epoch = epoch_ranged[-1]
            epoch_ranged.extend(range(last_epoch + 1, last_epoch + diff + 1))

    current_data = pd.DataFrame(
        {"epoch": epoch_ranged,
         "train_loss": training_losses,
         "val_losses": val_losses}
    )

    # load data if available
    file_path = os.path.join(__dir__, "plots", filename_of_logs)

    if not os.path.exists(file_path):
        current_data.to_csv(file_path, index=False)
        if __testing__:
            print("current data:\n", current_data.head())
    else:

        main_file = pd.read_csv(file_path)
        main_file = pd.concat([main_file, current_data], ignore_index=True)
        main_file.to_csv(file_path, index=False)

        if __testing__:
            print("main file:\n", main_file.head(5))
            print("current data:\n", current_data.head(5))


def save_model(epoch: int, model: torch.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.Any, filename: str, verbose: bool):
    save_path = os.path.join(__dir__, "saved_models", f"{
                             filename}_{epoch+1}_checkpoint.pt")
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, save_path)

    if verbose == True:
        Logger.info(f"Saved {save_path} Successfully...")


if __name__ == "__main__":

    args = sys.argv

    print(args)

    if args[1] == "--test-save-losses":

        __testing__ = True
        epoch = 20
        epoch_range = 5
        training_losses = [0]*5
        val_losses = [0]*5
        save_losses(epoch=epoch, training_losses=training_losses, filename="test",
                    val_losses=val_losses, epoch_range=epoch_range)
