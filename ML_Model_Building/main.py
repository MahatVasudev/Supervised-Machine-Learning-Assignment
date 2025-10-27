# model training
import torch
import torch.nn as nn
from datasets.fire_dataset import train_loader, val_loader
from models.CONVLSTM import ConvLSTM
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = nn.MSELoss()
model = ConvLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3)


def train_and_valid(epochs=10):
    print("Initializing Training....")
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        count = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            count += 1
            print("train set: ", count)
        model.eval()
        val_loss = 0
        print(f"Validating... Epoch {epoch+1}")
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                val_loss += criterion(y_pred, y).item()
        print(f"Validation Done...")

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print(f"Epoch [{
              epoch + 1}/{epochs}] | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")
        save_checkpoint(epoch, model, optimizer,
                        scheduler, filename="ConvLSTM")
        if epoch % 5 == 0:
            save_losses(epoch, train_losses, val_losses)

        if val_loss <= train_loss:
            save_checkpoint(epoch, model, optimizer,
                            scheduler, filename="ConvLSTM_Best")

    print("Training Done...")


def save_losses(epoch, train_losses, val_losses):
    n_loss = len(train_losses)
    df = pd.DataFrame({"epoch": list(range(1, n_loss+1)),
                      "train_loss": train_losses, "val_loss": val_losses})
    df.to_csv(f"./plots/till_{epoch+1}.csv", index=False)


def save_checkpoint(epoch, model, optimizer, scheduler, filename):
    save_path = f"./saved_models/{filename}_{epoch+1}_checkpoint.pt"
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, save_path)

    print(f"Saved {save_path} successfully")


if __name__ == '__main__':

    print('CUDA AVAILABLITY STATUS: ', torch.cuda.is_available())

    train_and_valid(epochs=20)
