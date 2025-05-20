import torch
import time
from argparse import Namespace

class EarlyStopping:
    def __init__(self, args: Namespace):
        self.patience = args.patience
        self.verbose = args.verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model, model_save_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_save_path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_save_path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), model_save_path)
        self.val_loss_min = val_loss


class EpochTimer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise ValueError(
                "Timer has not been started. Use start() to start the timer."
            )
        self.end_time = time.time()

    def reset(self):
        self.start_time = time.time()
        self.end_time = None

    def get_duration(self):
        if self.start_time is None:
            raise ValueError(
                "Timer has not been started. Use start() to start the timer."
            )
        if self.end_time is None:
            duration = time.time() - self.start_time
        else:
            duration = self.end_time - self.start_time
        if duration < 0:
            raise ValueError("Timer has not been stopped.")
        return duration

    def print_duration(self, epoch, total_epochs):
        duration = self.get_duration()
        print(f"Epoch {epoch}/{total_epochs} took {duration:.2f} seconds.")
