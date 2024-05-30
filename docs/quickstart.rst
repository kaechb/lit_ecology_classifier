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

1. **Clone the Repository**

   First, clone the repository to your local machine. You can skip this step if you have already downloaded the package.

   .. code-block:: bash

       git clone https://github.com/kaechb/lit_ecology_classifier.git
       cd lit_ecology_classifier

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies. Create and activate a virtual environment using `venv` or `conda`.

   Using `venv`:

   .. code-block:: bash

       python -m venv env
       source env/bin/activate  # On Windows, use `env\Scripts\activate`

   Using `conda`:

   .. code-block:: bash

       conda create --name lit_ecology python=3.8
       conda activate lit_ecology

3. **Install the Package**

   Install the package and its dependencies using `pip`.

   .. code-block:: bash

       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
       pip install .

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

Example
-------

Here's an example workflow:

1. Clone the repository and navigate to the project directory.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Prepare your dataset.
5. Run the training script.
6. Evaluate the model.
7. Generate and view the documentation.

Additional Resources
--------------------

- `PyTorch Documentation <https://pytorch.org/docs/stable/index.html>`_
- `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/>`_

Data Structure Details
----------------------

The `TarImageDataset` class expects the data to be structured as follows:

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

.. code-block:: bash

    tar -cvf dataset_name.tar dataset_name/
    # or
    zip -r dataset_name.zip dataset_name/
