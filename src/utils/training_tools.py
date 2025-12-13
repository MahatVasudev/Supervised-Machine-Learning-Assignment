import torch
import sys
import pandas as pd
import os
from src.config import BASE_DIR
from src.utils.logger import Logger

__testing__ = False


def save_losses(epoch, training_losses, val_losses, filename, epoch_range=5):
    filename_of_logs = f"{filename}_train_val_logs.csv"
    plot_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    file_path = os.path.join(plot_dir, filename_of_logs)

    # build epoch range
    epoch_ranged = list(range(epoch - epoch_range + 1, epoch + 1))

    # slice losses to match epoch_range length
    training_losses = training_losses[-epoch_range:]
    val_losses = val_losses[-epoch_range:]

    # construct dataframe
    current_data = pd.DataFrame({
        "epoch": epoch_ranged,
        "train_loss": training_losses,
        "val_losses": val_losses
    })

    if not os.path.exists(file_path):
        current_data.to_csv(file_path, index=False)
        if __testing__:
            print("current data:\n", current_data.head())
    else:
        main_file = pd.read_csv(file_path)
        main_file = pd.concat([main_file, current_data], ignore_index=True)
        # drop duplicate epochs, keep last
        main_file = main_file.drop_duplicates(subset="epoch", keep="last")
        main_file.to_csv(file_path, index=False)

        if __testing__:
            print("main file:\n", main_file.head(5))
            print("current data:\n", current_data.head(5))


def save_model(epoch: int, model, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.Any, filename: str, verbose: bool):
    save_path = os.path.join(BASE_DIR, "saved_models", f"{
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
