import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from social_tone.data.downloader import download_data


class MultimodalDataset(Dataset):
    def __init__(self, img_emb, txt_emb, labels):
        self.img_emb = torch.tensor(img_emb.values, dtype=torch.float32)
        self.txt_emb = torch.tensor(txt_emb.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"img": self.img_emb[idx], "txt": self.txt_emb[idx], "labels": self.labels[idx]}


class WarCovDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/raw",
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        # Проверка суммы долей
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1"

    def prepare_data(self):
        # Загружает данные через DVC API и выравнивает
        download_data(self.data_dir)

    def setup(self, stage=None):
        # Читаем уже выровненные локальные копии
        img_df = pd.read_csv(f"{self.data_dir}/multimodal_img_noft_pca.csv")
        txt_df = pd.read_csv(f"{self.data_dir}/multimodal_txt_noft_pca.csv")
        y_df = pd.read_csv(f"{self.data_dir}/multimodal_y_noft.csv")

        # Полный набор данных (все три датафрейма уже выровнены по длине)
        n_total = len(img_df)

        # --- 1. Сначала разделяем на train_val и test ---
        train_val_idx, test_idx = train_test_split(
            range(n_total), test_size=self.test_ratio, random_state=self.random_seed, shuffle=True
        )

        # --- 2. Затем train_val делим на train и val ---
        # Доля val от train_val
        val_size_relative = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size_relative, random_state=self.random_seed, shuffle=True
        )

        # --- 3. Создаём датасеты ---
        self.train_dataset = MultimodalDataset(
            img_df.iloc[train_idx].reset_index(drop=True),
            txt_df.iloc[train_idx].reset_index(drop=True),
            y_df.iloc[train_idx].reset_index(drop=True),
        )
        self.val_dataset = MultimodalDataset(
            img_df.iloc[val_idx].reset_index(drop=True),
            txt_df.iloc[val_idx].reset_index(drop=True),
            y_df.iloc[val_idx].reset_index(drop=True),
        )
        self.test_dataset = MultimodalDataset(
            img_df.iloc[test_idx].reset_index(drop=True),
            txt_df.iloc[test_idx].reset_index(drop=True),
            y_df.iloc[test_idx].reset_index(drop=True),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )
