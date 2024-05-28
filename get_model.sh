#!/bin/bash

# Define model name and URL of the checkpoint
MODEL_NAME="beitv2_base_patch16_224.in1k_ft_in22k_in1k"
URL="https://huggingface.co/timm/beitv2_base_patch16_224.in1k_ft_in22k_in1k/resolve/main/model.safetensors"
CHECKPOINT_DIR="checkpoints"

# Create the checkpoint directory if it doesn't exist
mkdir -p $CHECKPOINT_DIR

# Define the path to save the checkpoint
CHECKPOINT_PATH="$CHECKPOINT_DIR/${MODEL_NAME}.safetensors"

# Function to download the checkpoint
download_checkpoint() {
  if [ ! -f $CHECKPOINT_PATH ]; then
    echo "Downloading $MODEL_NAME checkpoint from Hugging Face..."
    wget -O $CHECKPOINT_PATH $URL
    if [ $? -eq 0 ]; then
      echo "Checkpoint saved to $CHECKPOINT_PATH"
    else
      echo "Failed to download the checkpoint!"
      exit 1
    fi
  else
    echo "Checkpoint already exists at $CHECKPOINT_PATH"
  fi
}

# Download the checkpoint
download_checkpoint
