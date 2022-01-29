# CodeTrek 
This is the official repository for [CodeTrek: Flexible Modeling of Code using an Extensible Relational
Representation](https://openreview.net/forum?id=WQc075jmBmf) accepted at ICLR'22.

Designing a suitable representation for code-reasoning tasks is challenging in aspects such as the kinds of program information to model, how to combine them, and how much context to consider. **CodeTrek** is a deep learning approach that addresses these challenges by representing codebases as databases that conform to rich relational schemas. The relational representation not only allows CodeTrek to uniformly represent diverse kinds of program information, but also to leverage program-analysis queries to derive new semantic relations, which can be readily incorporated without further architectural engineering. CodeTrek embeds this relational representation using a set of walks that can traverse different relations in an unconstrained fashion, and incorporates all relevant attributes along the way.

# Structure
This folder contains the following:
1. `data_prep`: code for constructing relational graphs from tables (`data_prep/graph`), code for generating biased random walks (`data_prep/random_walk`), and the tokenizer we used which is adapted from [CuBERT's tokenizer](https://github.com/google-research/google-research/tree/master/cubert) (`data_prep/tokenizer`).
2. `dbwalk/code2seq` and `dbwalk/ggnn`: two of the baselines that we implemented from scratch.
3. `python_codeql`: CodeQL queries that we used to construct the graphs.
4. `dbwalk/codetrek`: our implementation for training and testing codetrek models


## Structure of Data

The **codetrek data** for `dev`, `eval`, and `train` should be in separate folders. A file called `all_labels.txt` should contain the labels that are used for the task, each line one label. Each folder contains `n` graph files (`graph_*.gv`) and their corresponding stubs (`stub_*.json`) with information about the labels, anchors, etc. In the rest of this manual, we assume the following structure for the data directory.


```
dev
    \____ graph_file_1.py.gv
    \____ stub_file_1.py.json
    \____ ...
eval
    \____ graph_file_1.py.gv
    \____ stub_file_1.py.json
    \____ ...
train
    \____ graph_file_1.py.gv
    \____ stub_file_1.py.json
    \____ ...
all_labels.txt
```

The json structure that we have used for stubs can be found in `data_prep/random_walk/datapoint.py`.

After the data is structured in the desccribed format, the following steps are made to pre-process/train/test.


## Setup

Loading the graphs and generating the walks require `graphviz`.
So, first, install graphviz by running the following command:

    ./install_pygraphviz.sh

Then, install the required packages.

    pip install -e requirements.txt

If you wish to run the ggnn baseline, make sure to install `graphnet` by running `make` at `dbwalk/ggnn/graphnet`.
    
CodeTrek is implemented under `dbwalk/codetrek`. `[TASK]` is one of `var_def_use`, `var_misuse`, `ex_classify`, or `var_shadow`.

After setting the correct paths in `dbwalk/codetrek/[TASK]/run_cook_stub_gv.sh`, run it:

    ./dbwalk/codetrek/[TASK]/run_cook_stub_gv.sh

You should expect to see a set of chunked binary files under the cooked train/dev/test folders, as well as a dictionary pickle.


## Training CodeTrek

Configure the path/data name and other hyperparameters in `dbwalk/codetrek/[TASK]/dist_main.sh`, and run it.
You can run the scripts with and without gpu cores.

    ./dbwalk/codetrek/[TASK]/dist_main.sh

If your GPU memory is not large enough (V100 with 16G mem was used when developing this package), reduce the batch size accordingly.

To perform training without any gpu cores, remove `gpu_list` from the arguments inside the script and run the following command instead:

    ./dbwalk/codetrek/[TASK]/dist_main.sh -gpu -1


## Testing CodeTrek

Currently, the model dump name is `model-best_dev.ckpt` by default. So to evaluate, simply do:

    ./dbwalk/codetrek/[TASK]/dist_main.sh -phase eval -model_dump model-best_dev.ckpt


## Preparing data for code2seq/ggnn

First, set the variables in `run_cook_data.sh` and then run one of the following:

    ./dbwalk/code2seq/[TASK]/run_cook_data.sh
    ./dbwalk/ggnn/[TASK]/run_cook_data.sh

## Training code2seq/ggnn

First, set the variables in `run_cook_data.sh` and then run one of the following:

    ./dbwalk/code2seq/[TASK]/run_main.sh
    ./dbwalk/ggnn/[TASK]/run_main.sh

## Testing code2seq/ggnn

    ./dbwalk/code2seq/[TASK]/run_main.sh -phase eval -model_dump model-best_dev.ckpt
    ./dbwalk/ggnn/[TASK]/run_main.sh -phase eval -model_dump model-best_dev.ckpt
