import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class MultimodalClassifier(pl.LightningModule):
    def __init__(
        self,
        img_dim: int = 512,
        txt_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 50,
        lr: float = 1e-3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.dropout = dropout

        # Early fusion: конкатенация эмбеддингов
        self.classifier = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Метрики
        self.test_f1_micro = torchmetrics.F1Score(
            task="multilabel", num_labels=num_classes, average="micro"
        )
        self.test_precision = torchmetrics.Precision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.test_recall = torchmetrics.Recall(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.test_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        self.test_f1_macro = torchmetrics.F1Score(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.train_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        self.val_f1_macro = torchmetrics.F1Score(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.val_f1_micro = torchmetrics.F1Score(
            task="multilabel", num_labels=num_classes, average="micro"
        )
        self.val_precision = torchmetrics.Precision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.val_recall = torchmetrics.Recall(
            task="multilabel", num_labels=num_classes, average="macro"
        )

        # Потеря
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, img_emb, txt_emb):
        x = torch.cat([img_emb, txt_emb], dim=1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        img = batch["img"]
        txt = batch["txt"]
        labels = batch["labels"]

        logits = self.forward(img, txt)
        loss = self.criterion(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_acc(logits, labels)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        img = batch["img"]
        txt = batch["txt"]
        labels = batch["labels"]

        logits = self.forward(img, txt)
        loss = self.criterion(logits, labels)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        self.test_recall(logits, labels)
        self.test_precision(logits, labels)
        self.test_f1_micro(logits, labels)
        self.test_acc(logits, labels)
        self.test_f1_macro(logits, labels)

        self.log_dict(
            {"test_acc": self.test_acc, "test_f1_macro": self.test_f1_macro}, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        txt = batch["txt"]
        labels = batch["labels"]

        logits = self.forward(img, txt)
        loss = self.criterion(logits, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        # Логируем все метрики
        self.val_acc(logits, labels)
        self.val_f1_macro(logits, labels)
        self.val_f1_micro(logits, labels)
        self.val_precision(logits, labels)
        self.val_recall(logits, labels)

        self.log_dict(
            {
                "val_acc": self.val_acc,
                "val_f1_macro": self.val_f1_macro,
                "val_f1_micro": self.val_f1_micro,
                "val_precision": self.val_precision,
                "val_recall": self.val_recall,
            },
            on_epoch=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
