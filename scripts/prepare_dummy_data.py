from pathlib import Path

import numpy as np

dummy_img = np.random.randn(1, 512).astype(np.float32)
dummy_txt = np.random.randn(1, 768).astype(np.float32)

Path("data/dummy").mkdir(parents=True, exist_ok=True)
np.savez("data/dummy/dummy_input.npz", img=dummy_img, txt=dummy_txt)
print("✅ Пример данных сохранён в data/dummy/dummy_input.npz")
