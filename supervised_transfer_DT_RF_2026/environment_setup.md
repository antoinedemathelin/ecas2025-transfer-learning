# ADAPT Trees — environment setup (Conda or pip)

This guide provides **two independent installation options** for the supervised decision-tree / random-forest transfer-learning notebook:

- **Option A — Conda-first:** create a Conda environment, install scientific packages with Conda, then use pip for packages not reliably available through Conda (notably ADAPT and the pinned SciKeras release).
- **Option B — pip-first:** create a standard Python virtual environment and install Python packages with pip. Conda is not required.

**Choose one option**, not both. Both keep the notebook's dependencies isolated from your system Python and other environments.

## Recommended versions

| Component | Version |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.15.0 |
| SciKeras | 0.12.0 |
| ADAPT | 0.4.5 |
| NumPy | 1.26.2 |
| SciPy | 1.11.4 |
| scikit-learn | 1.3.2 |

These pins are intended to avoid incompatibilities between the ADAPT release and newer Keras/TensorFlow versions. The commands assume a supported machine/platform and access to package repositories.

---

## Option A — Conda-first (recommended if you already use Anaconda/Miniconda)

### A1. Create and activate a new environment

```bash
conda create -n adapt_trees python=3.10 pip -y
conda activate adapt_trees
```

### A2. Install scientific, plotting and Jupyter packages through Conda

```bash
conda install -c conda-forge \
    numpy=1.26.2 \
    scipy=1.11.4 \
    scikit-learn=1.3.2 \
    pandas \
    matplotlib \
    seaborn \
    cvxopt=1.3.2 \
    ipykernel \
    notebook \
    nbformat \
    nbclient \
    graphviz \
    python-graphviz \
    -y
```

### A3. Install ADAPT and its compatibility dependencies using pip **inside the active Conda environment**

```bash
python -m pip install \
    "tensorflow==2.15.0" \
    "scikeras==0.12.0" \
    "adapt==0.4.5"
```

**Why a pip step?** This is a **Conda-first hybrid installation**, not a claim that every dependency is distributed via Conda. ADAPT is provided on PyPI and the requested SciKeras version may not be found in your Conda channels. Using `python -m pip` after `conda activate adapt_trees` installs into **that environment only**. Do not run `conda install` again for the same pip-installed packages unless intentionally resolving dependencies.

### A4. Register the Jupyter kernel

```bash
python -m ipykernel install --user \
    --name adapt_trees \
    --display-name "Python (ADAPT Trees — Conda)"
```

### A5. Launch Jupyter

```bash
jupyter notebook
```

Select **Python (ADAPT Trees — Conda)** in the notebook's kernel menu.

For later sessions:

```bash
conda activate adapt_trees
jupyter notebook
```

---

## Option B — pip-first (no Conda needed)

**Prerequisite:** Install **Python 3.10** and ensure you can invoke it from a terminal. The example below uses `python3.10` on macOS/Linux, or `py -3.10` on Windows.

### B1. Create a Python virtual environment

**macOS / Linux:**

```bash
python3.10 -m venv .venv_adapt_trees
source .venv_adapt_trees/bin/activate
```

**Windows (PowerShell):**

```powershell
py -3.10 -m venv .venv_adapt_trees
.venv_adapt_trees\Scripts\Activate.ps1
```

### B2. Install pip tools and notebook dependencies

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
    "numpy==1.26.2" \
    "scipy==1.11.4" \
    "scikit-learn==1.3.2" \
    pandas matplotlib seaborn \
    cvxopt ipykernel notebook nbformat nbclient \
    graphviz \
    "tensorflow==2.15.0" \
    "scikeras==0.12.0" \
    "adapt==0.4.5"
```


---

## Verification

Run these commands **after activating the chosen environment**:

```bash
python -c "import tensorflow as tf; import sklearn; import adapt; print('TensorFlow:', tf.__version__); print('scikit-learn:', sklearn.__version__)"
```

```bash
python -c "from adapt.parameter_based import TransferTreeClassifier, TransferForestClassifier, TransferForestSelector; print('ADAPT tree and forest imports OK')"
```

## Remarks

- `cvxopt` is included for compatibility with broader ADAPT experiments; it is typically not required solely for the tree/forest transfer methods.
- Avoid updating TensorFlow/Keras independently without checking compatibility with this pinned ADAPT environment.
