from fastai.callback.core import Callback
import matplotlib.pyplot as plt


# Define a custom callback to capture training loss and plot it
class TrainingLogger(Callback):
    def __init__(self):
        self.losses = []
        self.epochs = []

    def after_epoch(self):
        # Capture loss and epoch number after each epoch
        self.losses.append(self.learn.loss.item())  # Get the loss value
        self.epochs.append(self.learn.epoch)

        print(f"Epoch {self.learn.epoch}: Loss = {self.learn.loss.item():.4f}")

    def after_fit(self):
        # Plot the loss over time after training
        plt.plot(self.epochs, self.losses)
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


