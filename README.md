# LightWeight_rgbd_network
This  is the code of a light weight network.

## prepare dataset
You can download the [SUNRGBD](http://rgbd.cs.princeton.edu/data/SUNRGBD.zip) by click here.
and use `LDN_SUNRGBD.py` to build the dataset

- `phase_train` : use train dataset or test  dataset
- `data_dir` ： the data path of SUNRGBD,
- `transform` : some data transform,you can find them in `LDN_transforms.py`

> new the dir `LDN_data_*`  befor run `LDN_*.py`
