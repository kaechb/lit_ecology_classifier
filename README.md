# Lit Ecology Classifier

Lit Ecology Classifier is a machine learning project designed for image classification tasks. It leverages PyTorch Lightning for streamlined training and evaluation processes.

## Features

- Easy configuration and setup
- Utilizes PyTorch Lightning for robust training and evaluation
- Supports training on multiple GPUs
- Test Time Augmentation (TTA) for enhanced evaluation
- Integration with Weights and Biases for experiment tracking

## Installation

To install Lit Ecology Classifier, use pip:

```bash
pip install lit-ecology-classifier
```

## Usage

### Training

To train the model, use the following command:

```bash
python -m lit_ecology_classifier.main --max_epochs 20 --dataset phyto --priority config/priority.json
```

### Inference

To run inference on unlabelled data, use the following command:

```bash
python -m lit_ecology_classifier.predict --datapath /path/to/data.tar --model_path /path/to/model.ckpt --outpath ./predictions/
```

## Configuration

The project uses an argument parser for configuration. Here are some of the key arguments:

### Training Arguments

- `--datapath`: Path to the tar file containing the training data.
- `--train_outpath`: Output path for training artifacts.
- `--main_param_path`: Main directory where the training parameters are saved.
- `--dataset`: Name of the dataset.
- `--use_wandb`: Use Weights and Biases for logging.
- `--priority_classes`: Path to the JSON file with priority classes.
- `--balance_classes`: Balance the classes for training.
- `--batch_size`: Batch size for training.
- `--max_epochs`: Number of epochs to train.
- `--lr`: Learning rate for training.
- `--lr_factor`: Learning rate factor for training of full body.
- `--no_gpu`: Use no GPU for training.

### Inference Arguments

- `--outpath`: Directory where predictions are saved.
- `--model_path`: Path to the model file.
- `--datapath`: Path to the tar file containing the data to classify.
- `--no_gpu`: Use no GPU for inference.
- `--no_TTA`: Disable test-time augmentation.

## Documentation

Detailed documentation for this project is available at [Read the Docs](https://lit-ecology-classifier.readthedocs.io).

### Example SLURM Job Submission Script

Here is an example SLURM job submission script for training on multiple GPUs:

```bash
#!/bin/bash
#SBATCH --account="em09"
#SBATCH --constraint='gpu'
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
export OMP_NUM_THREADS=12 #$SLURM_CPUS_PER_TASK
cd ${SCRATCH}/lit_ecology_classifier
module purge
module load daint-gpu cray-python
source lit_ecology/bin/activate
python -m lit_ecology_classifier.main --max_epochs 2 --dataset phyto --priority config/priority.json
```
