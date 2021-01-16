# relational-representation


# dbwalk


## Setup

Please install the package first. At the root folder of the project, do 

    pip install -e .
    
## Data preparation

First organize the dataset in the following format. The dataset is split into **train/dev/test**

```
walks
|___train
|   |__xx.json
|   |__yy.json
|   |__...
|
|___dev
|   |__...
|
|___test
|   |__...
|
```

Then navigate to `dbwalk/var_def_use`, and configure `run_cook_data.sh` appropriately, and do

    ./run_cook_data.sh
    
You should expect to see a set of chunked binary files under the cooked train/dev/test folders, as well as a dictionary pickle. 


## Training

Configure the path/data name correctly in `dbwalk/var_def_use/run_main.sh`, and simply run it for a while. 
You don't need to run till the end. Just stop it when the dev performance plateaus. 

```
cd dbwalk/var_def_use
./run_main.sh
```

If your GPU memory is not large enough (V100 with 16G mem was used when developing this package), reduce the batch size correspondinly, or feel free to change other model configurations.

## Test

Right now the model dump name is `model-best_dev.ckpt` by default. So to evaluate, simply do:

```
cd dbwalk/var_def_use
./run_main.sh -phase test -model_dump model-best_dev.ckpt
```
