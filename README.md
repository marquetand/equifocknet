# EquiFockNet

This repository contains the code and Jupyter Notebooks for predicting symmetric, physically accurate Fock matrices using $SO(3)$ equivariant neural networks. It includes a customized, natively integrated version of PhiSNet.

## 1. Project Setup

This guide will walk you through setting up the environment, getting the data, and running the project. The repository is designed to be fully self-contained.

### 1.1. Code Installation

Clone the repository to your local machine:

```bash
git clone [https://github.com/marquetand/equifocknet.git](https://github.com/marquetand/equifocknet.git)
cd equifocknet
```

### 1.2. Directory Structure

The repository uses an automated directory structure to keep code, data, and outputs organized. When you run the notebook for the first time, it will automatically generate the following ignored folders:

* `data_electrocyclic/`: Place your raw molecular data here. It will also hold the compiled SQLite and `.npz` datasets.
* `training_output/`: Contains all model checkpoints, TensorBoard logs, and `.pt` model weights.
* `analysis/`: Contains all generated visualization plots (e.g., `.png` files).
* `phisnet_fork/`: The customized core neural network architecture.

### 1.3. Data Preparation

**[Note: Data source to be specified]**

You will need to download the project dataset containing the Molcas calculations. Once downloaded, extract the files directly into the `data_electrocyclic/open/` directory.

### 1.4. Environment Setup

It is highly recommended to use Conda to manage the Python environment. If you do not have Conda installed on your machine, please download and install [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) first.

**1. Create and activate a new Conda environment:**
```bash
conda create -n equifocknet python=3.13
conda activate equifocknet
```

**2. Install `uv` and project dependencies:**
We use `uv`, an extremely fast Python package installer, to handle all dependencies (including PyTorch). First, install `uv`:
```bash
pip install uv
```

Next, install the project requirements. **Please choose the correct command below depending on your hardware:**

* **Option A: You have an NVIDIA GPU (Recommended)**
  This command points `uv` to the PyTorch index that contains the hardware-accelerated CUDA 11.8 bindings:
  ```bash
  uv pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
  ```

* **Option B: You DO NOT have an NVIDIA GPU (CPU Only or macOS)**
  This command points `uv` to the default CPU-only PyTorch index:
  ```bash
  uv pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
  ```

## 2. Running the Model

Once your environment is set up and the data is placed in `data_electrocyclic/open/`, you can launch the Jupyter Notebook:

```bash
jupyter notebook equifocknet.ipynb
```

The notebook is configured to automatically route all data reading and model saving to the correct internal folders. You can seamlessly switch between overfitting tests and full production runs by adjusting the `N_TRAIN_SAMPLES` variable in the first cell of the notebook.