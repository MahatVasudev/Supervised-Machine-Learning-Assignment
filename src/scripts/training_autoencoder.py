import torch
import gc
import torch.nn as nn
from src.utils.argparser import parser as training_parser
from src.utils.logger import Logger
from src.utils.training_tools import save_model, save_losses
from src.models.Auto_Encoder import AutoEncoderFire
from src.datasets.fire_dataset import FireGridAutoEncoderDataset
from src.datasets.context import DEFAULT_AUTOENCODER_DATA_CONFIG, FireDatasetContext
# INFO: Training Auto Encoder

# CONSTANTS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

args = training_parser.parse_args()

EPOCHS = args.epochs
FILENAME = args.filename
VERBOSE_MODE = args.verbose
LOG_LOSS = args.log_loss
LR = args.lr
WEIGHT_DECAY = args.weight_decay
SCHEDULER_PATIENCE = args.scheduler_patience

# Model
model = AutoEncoderFire(in_channel=1, bottleneck_channel=256,
                        hidden_channels=[64, 128, 256]).to(DEVICE)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE)
criterion = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')
data_context = FireDatasetContext(
    FireGridAutoEncoderDataset, DEFAULT_AUTOENCODER_DATA_CONFIG)

ae_train_loader, ae_val_loader, _ = data_context.load_dataloader()


def training_autoencoder():
    Logger.info("Training Autoencoder")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(0, EPOCHS):
        model.train()
        train_loss = 0
        for X, y in ae_train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                y_pred = model(X).to(DEVICE)
                loss = criterion(y_pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X, y in ae_val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                y_pred = model(X).to(DEVICE)
                val_loss += criterion(y_pred, y).item()

        val_loss /= len(ae_val_loader)
        train_loss /= len(ae_train_loader)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if VERBOSE_MODE:
            Logger.info_line(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {
                train_loss} | Val Loss: {val_loss} | LR: {scheduler.get_last_lr()[0]} ")

        if LOG_LOSS:
            if (epoch + 1) % 5 == 0:
                save_losses(epoch, training_losses=train_losses,
                            val_losses=val_losses, filename=FILENAME, epoch_range=5)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_model(epoch, model, optimizer=optimizer,
                       scheduler=scheduler, filename=f"{FILENAME}_Best", verbose=VERBOSE_MODE)

        if (epoch + 1) % 10 == 0:
            if VERBOSE_MODE:
                Logger.warning_line("Cleaning Garbage...")
            with torch.no_grad():
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    Logger.info(f"USING DEVICE: {DEVICE}")
    data_context.summary()
    training_autoencoder()

    Logger.executed("Training Done")
