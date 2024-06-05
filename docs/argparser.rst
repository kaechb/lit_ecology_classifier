ArgumentParser
=================================

This module provides functions to create argument parsers for configuring, training, and running the image classification model.

### `argparser()`

```eval_rst
.. autofunction:: argparser
   :noindex:
```

Creates an argument parser for configuring and training the machine learning model.

**Arguments:**

* `--datapath` (str): Path to the tar file containing the training data. Default is "/store/empa/em09/aquascope/phyto.tar".
* `--train_outpath` (str): Output path for training artifacts. Default is "./train_out".
* `--main_param_path` (str): Main directory where the training parameters are saved. Default is "./params/".
* `--dataset` (str): Name of the dataset. Default is "phyto".
* `--use_wandb` (flag): Use Weights and Biases for logging. Default is False.
* `--priority_classes` (str): Path to the JSON file specifying priority classes for training. Default is an empty string.
* `--rest_classes` (str): Path to the JSON file specifying rest classes for training. Default is an empty string.
* `--balance_classes` (flag): Balance the classes for training. Default is False.
* `--batch_size` (int): Batch size for training. Default is 180.
* `--max_epochs` (int): Number of epochs to train. Default is 20.
* `--lr` (float): Learning rate for training. Default is 1e-2.
* `--lr_factor` (float): Learning rate factor for training of full body. Default is 0.01.
* `--no_gpu` (flag): Use no GPU for training. Default is False.
* `--testing` (flag): Set this to True if in testing mode, False for training. Default is False.
* `--loss` (str): Loss function to use (choices: "cross_entropy", "focal"). Default is "cross_entropy".
* `--no_TTA` (flag): Enable Test Time Augmentation. Default is False.


**Returns:**

`argparse.ArgumentParser`: The argument parser with defined arguments.



### `inference_argparser()`

```eval_rst
.. autofunction:: inference_argparser
   :noindex:
```

Creates an argument parser for using the classifier on unlabeled data.

**Arguments:**

* `--batch_size` (int): Batch size for inference. Default is 180.
* `--outpath` (str): Directory where predictions will be saved. Default is "./preds/".
* `--model_path` (str): Path to the model checkpoint file. Default is "./checkpoints/model.ckpt".
* `--datapath` (str): Path to the tar file containing the data to classify. Default is "/store/empa/em09/aquascope/phyto.tar".
* `--no_gpu` (flag): Use no GPU for inference. Default is False.
* `--no_TTA` (flag): Disable test-time augmentation. Default is False.
* `--gpu_id` (int): GPU ID to use for inference. Default is 0.
* `--limit_pred_batches` (int): Limit the number of batches to predict. Default is 0, meaning no limit, set a low number to debug.
* `--prog_bar` (flag): Enable progress bar. Default is False.

**Returns:**

`argparse.ArgumentParser`: The argument parser with defined arguments.



**Example Usage:**

```python
if __name__ == "__main__":
    parser = argparser()  # or inference_argparser()
    args = parser.parse_args()
    print(args)
```
