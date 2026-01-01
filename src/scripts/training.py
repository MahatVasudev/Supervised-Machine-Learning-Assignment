import gc
import os

import pytorch_msssim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import MODEL_DIR
from datasets.context import DEFAULT_CONVLSTM_DATA_CONFIG, FireDatasetContext
from datasets.fire_dataset import FireSpreadDatasetLazy
from models.Auto_Encoder import AutoEncoderFire
from models.CONVLSTM_NEW import CONVLSTM_FIREMODEL
# from models.CONVLSTM import ConvLSTM, ConvLSTM2Layers
from utils.argparser import parser as training_parser
from utils.logger import Logger
from utils.training_tools import save_losses, save_model

device = "cuda" if torch.cuda.is_available() else "cpu"

args = training_parser.parse_args()

# INFO: All of the command line arguments

EPOCHS = args.epochs
FILENAME = args.filename
VERBOSE_MODE = args.verbose
LOG_LOSS = args.log_loss
LR = args.lr
WEIGHT_DECAY = args.weight_decay
SCHEDULER_PATIENCE = args.scheduler_patience


auto_encoder_model = AutoEncoderFire(
    in_channel=1, hidden_channels=[64, 128, 256], bottleneck_channel=256
)
auto_encoder_model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "AUTOENCODER_VICTORY_BIGGER_MORE_EPOCHS_BETTER_Best_94_checkpoint.pt",
        )
    )["model"]
)

for p in auto_encoder_model.encoder_channels.parameters():
    p.requires_grad = False

for k in auto_encoder_model.decoder_channels.parameters():
    k.requires_grad = True

auto_encoder_model.train()

model = CONVLSTM_FIREMODEL(auto_encoder=auto_encoder_model, hidden_channels=256).to(
    device
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE
)
mse_loss = nn.MSELoss()
scaler = torch.amp.GradScaler("cuda")


def training(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs=EPOCHS,
):
    print("Initializing Training...")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda"):
                y_pred = model(X).to(device)
                loss = combined_loss(y_pred, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # validation evaluation

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X).to(device)
                val_loss += combined_loss(y_pred, y).item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if VERBOSE_MODE:
            Logger.info_line(
                f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {
                    train_loss} | Val Loss: {val_loss} | LR: {scheduler.get_last_lr()[0]} "
            )

        if LOG_LOSS:
            if (epoch + 1) % 5 == 0:
                save_losses(
                    epoch,
                    training_losses=train_losses,
                    val_losses=val_losses,
                    filename=FILENAME,
                    epoch_range=5,
                )

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_model(
                epoch,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                filename=f"{FILENAME}_Best",
                verbose=VERBOSE_MODE,
            )

        if (epoch + 1) % 10 == 0:
            if VERBOSE_MODE:
                Logger.warning_line("Cleaning Garbage...")
            with torch.no_grad():
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()


def combined_loss(y_pred, y_true):
    return mse_loss(y_pred, y_true) + 0.5 * (
        1 - pytorch_msssim.ssim(y_pred, y_true, data_range=1.0)
    )


if __name__ == "__main__":
    print(EPOCHS)

    data_context = FireDatasetContext(
        dataset=FireSpreadDatasetLazy, config=DEFAULT_CONVLSTM_DATA_CONFIG
    )

    train_loader, val_loader, _ = data_context.load_dataloader()
    data_context.summary()
    training(train_loader=train_loader, val_loader=val_loader, epochs=EPOCHS)
    # TODO: Add a plotting function of metrics at the end
    Logger.executed("Training Completed! Rejoice!!!")
