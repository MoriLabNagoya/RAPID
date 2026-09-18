# RAPID

An updated version of the RootMeasurement tool.

## Current segmentation pipeline

The legacy Keras/TensorFlow U-Net path has been removed. RAPID now uses:

- YOLO for ROI detection (`best.pt`)
- `ICML-NoAI` for the legacy classical segmentation path
- `ICML-improve` as the default segmentation path

No `keras_segmentation`, Keras, TensorFlow, TensorBoard, or `ml-dtypes` installation is required.

## Recommended Windows environment

RAPID is configured for Python 3.10.20.

### Create from `environment.yml`

```bat
conda env create -f environment.yml
conda activate rapid
python Application.py
```

### Or create the environment manually

```bat
conda create -n rapid python=3.10.20 -y
conda activate rapid
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
python Application.py
```

If an older `rapid` environment contains incompatible NumPy/scikit-image binaries, recreating the environment is recommended instead of installing over the old environment.
