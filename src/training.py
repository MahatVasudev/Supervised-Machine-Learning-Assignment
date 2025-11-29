import torch
import gc
import torch.nn as nn
from datasets.fire_dataset import train_loader, val_loader
from models.CONVLSTM import ConvLSTM, ConvLSTM2Layers
import pandas as pd
import sys
from utils.argparser import parser as training_parser
from utils.logger import Logger
from utils.training_tools import save_model, save_losses

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = training_parser.parse_args()

# INFO: All of the command line arguments

EPOCHS = args.epochs
FILENAME = args.filename
VERBOSE_MODE = args.verbose
LOG_LOSS = args.log_loss
LR = args.lr
WEIGHT_DECAY = args.weight_decay
SCHEDULER_PATIENCE = args.scheduler_patience


model = ConvLSTM().to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE)
criterion = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler()


def training(epochs=EPOCHS):
    print("Initializing Training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():

                y_pred = model(X)
                loss = criterion(y_pred, y)

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
                y_pred = model(X)
                val_loss += criterion(y_pred, y).item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if VERBOSE_MODE:
            Logger.info_line(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {
                train_loss:.5f} | Val Loss: {val_loss} | LR: {scheduler.get_last_lr()[0]} ")

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
                if device == "cuda":
                    torch.cuda.empty_cache()
