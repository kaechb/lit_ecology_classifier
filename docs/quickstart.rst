Quickstart Guide
================

Welcome to the quickstart guide for `lit_ecology_classifier`! This guide will walk you through the installation process and show you how to use the package for ecological classification.

Prerequisites
-------------

Before you begin, ensure you have the following installed:

- Python 3.6 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

Installation
------------

# Lit Ecology Quickstart on Daint-GPU (CSCS)

This guide will help you quickly set up the Lit Ecology classifier on the Daint-GPU system at CSCS.

## Prerequisites

* Access to the Daint-GPU system (CSCS).
* Basic knowledge of Python environments and module systems.

## Steps

1. **Navigate to your scratch space:**

```bash
cd $SCRATCH
```

2. **Load necessary modules:**

```bash
module load daint-gpu cray-python
```

3. **Create a Python virtual environment:**

```bash
python -m venv lit_ecology
source lit_ecology/bin/activate
```

4. **Source the model script (replace with the actual path):**

```bash
source get_model.sh
```

5. **Create directories for parameters and phyto data:**

```bash
mkdir params
mkdir params/phyto
```

6. **Upgrade pip and install PyTorch:**

```bash
lit_ecology/bin/python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

7. **Install lit_ecology_classifier and timm:**

```bash
pip install lit_ecology_classifier
pip install timm==0.9.2
```

8. **Create a directory for Slurm scripts:**

```bash
mkdir slurm
```


## Important Notes:

* **Replace placeholders:** Replace the placeholders (e.g., `/store/empa/...`) with the actual paths to your files and directories.
* **GPU version:** Make sure that the version of the cudatoolkit installed in the venv matches the version of the GPU on the cluster (here cu118).
* **Slurm scripts:** You'll likely need to create Slurm scripts in the `slurm` directory to run your jobs efficiently on the cluster. Refer to the official CSCS documentation for guidance on writing Slurm scripts.

Usage
-----

1. **Prepare Your Data**

   Ensure your dataset is structured in the required format. The data should be organized with each class having its own subdirectory. The overall structure should look like this:

   .. code-block:: bash

       dataset_name/
       ├── class1/
       │   ├── image1.jpg
       │   ├── image2.jpg
       │   └── ...
       ├── class2/
       │   ├── image1.jpg
       │   ├── image2.jpg
       │   └── ...
       └── ...

   Once organized, compress the dataset into a `.tar` or `.zip` file.

2. **Train the Model**

   To train the model, run the following command with appropriate arguments:

   .. code-block:: bash

       python -m lit_ecology_classifier.main --max_epochs 2 --dataset phyto --priority config/priority.json

   **Arguments**:
   - `--max_epochs`: The number of epochs to train.
   - `--dataset`: The name of the dataset.
   - `--priority`: Path to the priority configuration file.

3. **Evaluate the Model**

   After training, you can evaluate the model on your test dataset. Modify the script as necessary to point to your test data.

   .. code-block:: bash

       python -m lit_ecology_classifier.evaluate --dataset phyto --priority config/priority.json

4. **Generate Documentation**

   To generate the documentation, navigate to the `docs` directory and run:

   .. code-block:: bash

       cd docs
       make html

   You can view the generated documentation by opening `docs/_build/html/index.html` in your web browser.

Additional Resources
--------------------

- `PyTorch Documentation <https://pytorch.org/docs/stable/index.html>`_
- `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/>`_

Data Structure Details
----------------------

The `TarImageDataset` or `ImageFolderDataset` class expects the data to be structured as follows:

- The root directory should contain subdirectories for each class.
- Each subdirectory should contain the image files for that class.

Example structure:

.. code-block:: bash

    dataset_name/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...

Compress the `dataset_name` directory into a `.tar` or `.zip` file before using it with the `lit_ecology_classifier`.
The code will automatically deduce form the datapath whether it is .tar or folder based dataset.