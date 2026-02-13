"""
Data downloader using DVC API.
No manual 'dvc pull' required — DVC fetches data automatically.
"""

from pathlib import Path
from typing import Dict

import dvc.api
import pandas as pd


def download_data(data_path: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Load WarCov multimodal dataset via DVC API.

    Returns:
        dict: {'img': DataFrame, 'txt': DataFrame, 'y': DataFrame}
    """
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    files = {
        "img": "multimodal_img_noft_pca.csv",
        "txt": "multimodal_txt_noft_pca.csv",
        "y": "multimodal_y_noft.csv",
    }

    dataframes = {}
    for key, filename in files.items():
        with dvc.api.open(
            path=f"data/raw/{filename}",
            repo=".",
            mode="rb",
        ) as fd:
            df = pd.read_csv(fd)
            dataframes[key] = df
            local_path = data_path / filename
            df.to_csv(local_path, index=False)
            print(f"✅ Loaded {filename}: shape {df.shape}")

    min_len = min(len(dataframes["img"]), len(dataframes["txt"]), len(dataframes["y"]))

    for key in dataframes:
        dataframes[key] = dataframes[key].iloc[:min_len].reset_index(drop=True)
        print(f"   {key}: new shape {dataframes[key].shape}")

    return dataframes


if __name__ == "__main__":
    dfs = download_data()
    for key, df in dfs.items():
        print(f"{key}: {df.shape}")
