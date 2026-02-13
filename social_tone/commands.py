import subprocess

import hydra
import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from social_tone.data.datamodule import WarCovDataModule
from social_tone.models.classifier import MultimodalClassifier


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    # --- MLflow Logger ---
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
    )

    # Логируем версию Git-коммита
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        mlflow_logger.log_hyperparams({"git_commit": commit_hash})
    except Exception:
        mlflow_logger.log_hyperparams({"git_commit": "unknown"})

    # Логируем весь конфиг
    mlflow_logger.log_hyperparams(cfg)

    # --- DataModule ---
    dm = WarCovDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        random_seed=cfg.data.random_seed,
    )

    # --- Model ---
    model = MultimodalClassifier(
        img_dim=cfg.model.img_dim,
        txt_dim=cfg.model.txt_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
        dropout=cfg.model.dropout,
    )

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=cfg.training.early_stop_patience, mode="min"
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=cfg.logging.log_every_n_steps,
        deterministic=True,
    )

    trainer.fit(model, dm)

    if cfg.training.test_after_training:
        # Загружаем лучший чекпоинт
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            best_model = MultimodalClassifier.load_from_checkpoint(best_model_path)
            trainer.test(best_model, datamodule=dm)
        else:
            trainer.test(model, datamodule=dm)

    # --- Сохранение графиков в plots/ ---
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    from mlflow.tracking import MlflowClient

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    client = MlflowClient()
    run_id = mlflow_logger.run_id

    # Скачиваем историю метрик
    metrics = client.get_metric_history(run_id, "val_loss")
    if metrics:
        df = pd.DataFrame([(m.step, m.value) for m in metrics], columns=["step", "val_loss"])
        plt.figure()
        plt.plot(df["step"], df["val_loss"])
        plt.title("Validation Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(plots_dir / "val_loss.png")
        mlflow.log_artifact(str(plots_dir / "val_loss.png"))
        print("✅ Сохранён график val_loss.png")

    # Аналогично для других метрик
    for metric in ["val_acc", "val_f1_macro", "val_precision", "val_recall"]:
        metrics = client.get_metric_history(run_id, metric)
        if metrics:
            df = pd.DataFrame([(m.step, m.value) for m in metrics], columns=["step", metric])
            plt.figure()
            plt.plot(df["step"], df[metric])
            plt.title(metric.replace("_", " ").title())
            plt.xlabel("Step")
            plt.ylabel(metric)
            plt.savefig(plots_dir / f"{metric}.png")
            mlflow.log_artifact(str(plots_dir / f"{metric}.png"))


if __name__ == "__main__":
    train()
