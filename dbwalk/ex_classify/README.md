You can follow the following steps to reproduce the current best results we get. 

## data

The best result so far is obtained using func-level graphs built in Jan, 2021. 

The raw data is here:
https://drive.google.com/file/d/1LnUL6MA-aWxrRaZWehcoQP-FE_OA3tLq/view?usp=drive_web

and the cooked data can be found here:

https://drive.google.com/file/d/1akpGbEWdAklxrH-c5wLeJZDi9u4Zh8-q/view?usp=sharing&resourcekey=0-v7yR0W_D2mOmeqI0EY8-EQ


## train & eval

Firstly please download the cooked data and untar it to `$HOME/data/dataset/dbwalk`. Or if you want to place it somewhere else, please modify the scripts correspondingly.

### train

just do `./run_main.sh`. You can stop the training after ~40k steps of training. 
Remember to modify the data path and save dir according to your need.

### eval

Please run
```
  ./run_main.sh -phase eval -model_dump model-best_dev.ckpt
```

Currently the default configuration should give you ~60.5% accuracy. 

=======May 1 update======

The new script with distributed training gets 61.94% test accuracy on the exception_januray version.
However the same code on the exception_apr29 version only gets ~57%.
