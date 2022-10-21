Install requirements
```
pip install -r requirements.txt

```

# HOW TO TRAIN

1. Choose model and configuration from `scripts` folder
1. Add path to the root of the Cityscapes dataset in the json config file in variable `db_root`
1. Run training using the `run.sh` script with the folder name as first parameter and the config file name as second parameter

Examples:
```
./run.sh deeplab_base config_50.json
```
```
./run.sh swiftnet_base config.json

```

# HOW TO EVALUTATE

1. Check model and configuration of trained model from `scripts` folder
1. Run evaluation using the `run.sh` script with the folder name as first parameter and the config file name as second parameter, and the path to the learned model as third parameter

Example:
```
./run.sh deeplab_base config_50.json runs/deeplab/dl_50_sgd/experiment_0/checkpoint.pth

```
