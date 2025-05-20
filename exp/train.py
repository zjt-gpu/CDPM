import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import Namespace
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from utils.tools import EarlyStopping, EpochTimer
import time
from thop import profile

@dataclass
class PerformanceMetrics:
    mse_loss: float = 0.0
    mae_loss: float = 0.0

    def __repr__(self):
        return f"MSE Loss: {self.mse_loss:.6f}, MAE Loss: {self.mae_loss:.6f}"


class ModelTrainer:
    def __init__(
        self,
        args: Namespace,
        model: Namespace,
        device: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        test_dataset: Namespace
    ):
        self.args = args
        self.verbose = args.verbose
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_dataset = test_dataset

        # Loss functions
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        # Optimizer and Scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=args.lr_decay)

        # Training parameters
        self.num_epochs = args.num_epochs
        self.eval_freq = args.eval_frequency

        # Paths
        self.save_dir = Path(args.save_dir)
        self.train_log_path = self.save_dir / "train_log.txt"
        self.val_log_path = self.save_dir / "val_log.txt"
        self.test_log_path = self.save_dir / "test_log.txt"
        self.model_save_path = self.save_dir / "model_checkpoint.pth"

        self.early_stopping = EarlyStopping(self.args)
        self.epoch_timer = EpochTimer()

    def train(self):
        self.train_log_path.write_text("")
        self.val_log_path.write_text("")

        for epoch in range(self.num_epochs):
            self.epoch_timer.start()
            train_metrics, train_speed = self._train_one_epoch()

            with self.train_log_path.open("a") as log_file:
                log_file.write(f"Epoch {epoch + 1}: {train_metrics} | Speed: {train_speed:.2f} it/s\n")
            if self.verbose:
                print(f"Training Epoch {epoch + 1}: {train_metrics} | Speed: {train_speed:.2f} it/s")

            self.epoch_timer.stop()
            if self.verbose:
                self.epoch_timer.print_duration(epoch=epoch + 1, total_epochs=self.num_epochs)

            if (epoch + 1) % self.eval_freq == 0:
                val_metrics, val_speed = self._validate_one_epoch()

                with self.val_log_path.open("a") as log_file:
                    log_file.write(f"Epoch {epoch + 1}: {val_metrics} | Speed: {val_speed:.2f} it/s\n")
                if self.verbose:
                    print(f"Validation Epoch {epoch + 1}: {val_metrics} | Speed: {val_speed:.2f} it/s")

                self.early_stopping(val_metrics.mse_loss, self.model, self.model_save_path)
                if self.early_stopping.early_stop:
                    if self.verbose:
                        print("Early stopping triggered.")
                    break

    def _train_one_epoch(self):
        self.model.train()
        metrics = PerformanceMetrics()
        total_iters = 0
        total_time = 0

        for x, y in tqdm(self.train_loader, desc="Training", disable=not self.args.use_tqdm):
            start_time = time.time()
            self.optimizer.zero_grad()

            predictions = self.model(x, task="train")
            mse_loss = self.mse_criterion(predictions, y)
            mae_loss = self.mae_criterion(predictions, y)
            mse_loss.backward()

            self.optimizer.step()
            flops, params = profile(self.model, inputs=(x, "train"), verbose=False)

            total_time += time.time() - start_time
            total_iters += 1
            iter_speed = total_iters / total_time

            metrics.mse_loss += mse_loss.item()
            metrics.mae_loss += mae_loss.item()

        self.scheduler.step()

        metrics.mse_loss /= len(self.train_loader)
        metrics.mae_loss /= len(self.train_loader)
        avg_iter_speed = total_iters / total_time

        return metrics, avg_iter_speed

    @torch.no_grad()
    def _validate_one_epoch(self):
        self.model.eval()
        metrics = PerformanceMetrics()
        total_iters = 0
        total_time = 0

        for x, y in tqdm(self.val_loader, desc="Validation", disable=not self.args.use_tqdm):
            start_time = time.time()
            
            predictions = self.model(x, task="valid")
            mse_loss = self.mse_criterion(predictions, y)
            mae_loss = self.mae_criterion(predictions, y)

            total_time += time.time() - start_time
            total_iters += 1
            iter_speed = total_iters / total_time

            metrics.mse_loss += mse_loss.item()
            metrics.mae_loss += mae_loss.item()

        metrics.mse_loss /= len(self.val_loader)
        metrics.mae_loss /= len(self.val_loader)
        avg_iter_speed = total_iters / total_time

        return metrics, avg_iter_speed

    @torch.no_grad()
    def evaluate_test(self):
        self.test_log_path.write_text("")
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()

        metrics = PerformanceMetrics()
        for x, y in tqdm(self.test_loader, desc="Testing", disable=not self.args.use_tqdm):
            if self.args.loss_type == 'mse':
                predictions = self.model(x, task="test")
                mse_loss = self.mse_criterion(predictions, y)
                mae_loss = self.mae_criterion(predictions, y)

            metrics.mse_loss += mse_loss.item()
            metrics.mae_loss += mae_loss.item()

        metrics.mse_loss /= len(self.test_loader)
        metrics.mae_loss /= len(self.test_loader)

        with self.test_log_path.open("w") as log_file:
            log_file.write(f"{metrics}\n")

        return metrics
