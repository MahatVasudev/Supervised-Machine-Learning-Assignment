import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import BASE_DIR
import os


def plot_evaluation(csv_filename: str, model_name: str, addtional_title_context: str = ""):
    df = pd.read_csv(csv_filename)
    max_epoch = df["epoch"].max() + 1
    best_epoch = df[df['val_losses'] ==
                    df['val_losses'].min()]['epoch'].values + 1
    plt.suptitle(f"{model_name}")
    plt.title(f"Epochs: {max_epoch}: {addtional_title_context}")

    plt.plot(df['epoch']+1, df["train_loss"], 'r-o')
    plt.plot(df['epoch']+1, df["val_losses"], 'g-o')

    plt.axvline(best_epoch, c='blue')
    plt.ylim(0)

    plt.legend(["train loss", "validation loss", "Best Validation Loss"])
    plt.show()


if __name__ == "__main__":

    plot_name = os.path.join(
        BASE_DIR, "plots", "CONVLSTM_WITHAUTOENCODER_train_val_logs.csv")
    filename = "CONVLSTM_AUTOENCODER"
    additional_title_context = "Structure: 1->32->64->128-128-128<-64<-32<-1"

    plot_evaluation(plot_name, model_name=filename,
                    addtional_title_context=additional_title_context)
