# Social Tone Analyzer

## Постановка задачи
Разработка системы мультимодальной классификации постов из соцсетей (VK/Telegram) для автоматического определения тематики контента по 50 категориям (политика, экономика, пандемия, спорт и т.д.).
Система использует предобученные эмбеддинги изображений (ResNet-18) и текстов (Sentence Transformer), поверх которых обучается лёгкий MLP-классификатор.
Целевое применение — мониторинг инфополя и раннее обнаружение социально значимых событий.

## Структура репозитория
```
 social-tone-analyzer/
 ├── .dvc/
 ├── .gitignore
 ├── .pre-commit-config.yaml
 ├── README.md
 ├── data/
 │   ├── dummy/
 │   │   └── dummy_input.npz.dvc
 │   └── raw/
 │       ├── ...csv.dvc (3 файла)
 ├── models/
 │   └── triton_repo/
 │       └── multimodal_classifier/
 │           ├── 1/
 │           │   └── model.onnx
 │           └── config.pbtxt
 ├── plots/
 │   └── *.svg
 ├── poetry.lock
 ├── pyproject.toml
 ├── scripts/
 │   ├── convert_to_trt.sh
 │   ├── direct_onnx.py
 │   ├── export_onnx.py
 │   ├── prepare_dummy_data.py
 │   └── triton_client.py
 └── social_tone/
     ├── commands.py
     ├── configs/
     ├── data/
     ├── __init__.py
     └── models/
```
## Формат входных и выходных данных
**Входные данные:**
- Вектор эмбеддинга изображения: `np.ndarray`, shape `(512,)`, float32.
- Вектор эмбеддинга текста: `np.ndarray`, shape `(768,)`, float32.

*Примечание: в проекте используются уже извлечённые эмбеддинги из датасета WarCov, сырые картинки/текст не требуются.*

**Выходные данные:**
- Логиты для 50 классов: `np.ndarray`, shape `(50,)`, float32.
- После применения порога 0.5 получается бинарный вектор мультилейбла.

**Протокол взаимодействия:**
- CLI инференс через ONNX Runtime / PyTorch.
- REST/gRPC API через Triton Inference Server (опционально).

## Метрики
| Метрика       | Значение (val) | Комментарий                          |
|---------------|----------------|--------------------------------------|
| Accuracy      | ~0.95          | Доля полностью верных битовых векторов |
| Precision (macro)| ~0.57       | Точность (усреднение по классам)     |
| Recall (macro)| ~0.20          | Полнота (из-за сильного дисбаланса)  |
| F1-macro      | ~0.27          | Гармоническое среднее Precision/Recall|

Все метрики логируются в MLflow, доступны в UI.
Графики обучения (loss, accuracy, f1, precision, recall) сохраняются в папку [`plots/`](./plots) в формате PNG (экспорт из MLflow или скриншоты).

## Валидация и тест
**Стратегия разделения:**
- Train: 70%
- Validation: 15%
- Test: 15%

Разбиение выполняется **после выравнивания** всех трёх модальностей (img, txt, y) до одинакового количества строк (минимальное из трёх).
Фиксированный `random_seed=42` для воспроизводимости.

**Воспроизводимость:**
- Poetry lock, Hydra конфиги, версии в `pyproject.toml`.
- MLflow-трекинг каждого эксперимента.
- DVC для версионирования данных.

## Датасет
**WarCov Multimodal Dataset** (публичная версия с предвычисленными эмбеддингами).

**Характеристики:**
- ~87k мультимодальных постов (текст+изображение).
- 50 бинарных меток (автоматически по хэштегам).
- Эмбеддинги:
  - Изображения: ResNet-18 (ImageNet), **без дообучения**, 512 признаков + временные метки.
  - Текст: Sentence Transformer `clips/mfaq`, 768 признаков.

**Особенности:**
- Сильный дисбаланс классов (большинство меток — 0).
- Не все изображения прошли фильтрацию — после выравнивания остаётся **70 252** поста.

**Управление данными:**
- DVC + локальное хранилище.
- Автоматическая загрузка через `dvc.api.open()` в `downloader.py`.

## Моделирование
### Бейзлайн
Ранняя фузия: конкатенация эмбеддингов (512+768=1280) → Logistic Regression.
Качество: Accuracy ~0.93, F1-macro ~0.15.

### Основная модель
**Архитектура:**

[img_emb:512] + [txt_emb:768] → Concatenate (1280) → Linear(1280 → 256) → ReLU → Dropout(0.2) → Linear(256 → 50)

**Обучение:**
- Фреймворк: PyTorch Lightning.
- Оптимизатор: Adam, lr=0.001, ReduceLROnPlateau.
- Функция потерь: `BCEWithLogitsLoss`.
- Early stopping по `val_loss` (patience=5).
- Max эпох: 20.

**Аугментации:** не используются (работаем с фиксированными эмбеддингами).

## Внедрение
### ONNX
- Экспорт обученной модели в `models/multimodal_classifier.onnx`.
- Динамический batch size.
- Проверка через ONNX Runtime.

### TensorRT
- Скрипт `scripts/convert_to_trt.sh` конвертирует ONNX в TensorRT plan.
- Пример входных данных (`dummy_input.npz`) заверсионирован в DVC.

### Triton Inference Server
- Model repository: `models/triton_repo/multimodal_classifier`.
- Конфиг `config.pbtxt` для ONNX Runtime backend.
- Запуск через официальный минимальный образ Triton (`24.10-py3-min`).
- Клиент на Python (`triton_client.py`) отправляет запросы по HTTP.

---

## Setup

```bash
git clone https://github.com/zotovsem/social-tone-analyzer
cd social-tone-analyzer
poetry install
pre-commit install
```

Данные подтянутся автоматически при первом запуске обучения (через dvc.api).

## Train
Запусти MLflow сервер (для логов метрик):

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080
```
Запуск обучения (с конфигами Hydra):

```bash
poetry run python -m social_tone.commands
```
Можно переопределить параметры через командную строку:

```bash
poetry run python -m social_tone.commands training.max_epochs=5 model.lr=0.0005
```
Лучший чекпоинт сохраняется в checkpoints/.

## Logging and Plots
- Все эксперименты логируются в MLflow (локально по адресу http://127.0.0.1:8080).

- Графики метрик (loss, accuracy, f1, precision, recall) можно скачать из MLflow UI или найти готовые PNG в папке plots/.

- Версия кода (git commit hash) автоматически записывается в параметры эксперимента.

## Production Preparation
### 1. ONNX
```bash
poetry run python scripts/export_onnx.py
```
Результат: models/multimodal_classifier.onnx.

### 2. TensorRT (только NVIDIA GPU)
```bash
chmod +x scripts/convert_to_trt.sh
./scripts/convert_to_trt.sh
```
Создаёт models/multimodal_classifier.plan.
Пример данных для конверсии уже добавлен в DVC: data/dummy/dummy_input.npz.

### 3. Triton Model Repository
Репозиторий уже подготовлен в models/triton_repo/.
Запуск сервера:

```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models/triton_repo:/models \
  nvcr.io/nvidia/tritonserver:24.10-py3-min \
  tritonserver --model-repository=/models \
               --model-control-mode=explicit \
               --load-model=multimodal_classifier
```
## Infer
### CLI инференс (ONNX Runtime, без сервера)
```bash
poetry run pip install onnxruntime
poetry run python scripts/onnx_direct.py
```
Скрипт генерирует случайные эмбеддинги и выводит форму выходных логитов.
Для своих данных подставь .npy файлы с эмбеддингами.

### Инференс через Triton
Предварительно запусти сервер (см. выше).
Клиент:

```bash
poetry run pip install tritonclient[http]
poetry run python scripts/triton_client.py
```
Формат входных данных для Triton:

img_emb: тензор FP32, форма [batch, 512].

txt_emb: тензор FP32, форма [batch, 768].

Пример запроса (Python): см. scripts/triton_client.py.

Формат ответа:

logits: тензор FP32, форма [batch, 50].
