class EarlyStopping:
    """
    EarlyStopping monitors the validation loss during training and stops the training process
    if the loss does not improve after a specified number of epochs (patience).

    Attributes:
    - patience: The number of epochs to wait for improvement before stopping.
    - min_delta: Minimum change in the monitored quantity to qualify as an improvement.
    - counter: Counts the number of epochs without improvement.
    - best_loss: Stores the best validation loss encountered during training.
    - stopped_epoch: Records the epoch number when training was stopped.
    """

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.stopped_epoch = 0

    def __call__(self, val_loss):
        """
        Checks the validation loss to determine if training should stop.

        Parameters:
        - val_loss: The current validation loss to check against the best loss.

        Returns:
        - bool: True if training should stop, otherwise False.
        """
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if there's improvement
        else:
            self.counter += 1  # Increment counter if no improvement

        if self.counter >= self.patience:
            self.stopped_epoch = self.counter
            return True  # Stop training

        return False  # Continue training
