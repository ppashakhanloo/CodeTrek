# Global Relational Models of Source Code
This repository contains the data and code to replicate our [ICLR 2020 paper](http://vhellendoorn.github.io/PDF/iclr2020.pdf) on models of source code that combine global and structural information, including the Graph-Sandwich model family and the GREAT (Graph-Relational Embedding Attention Transformer) model.

## Pre-trained Models
In response to multiple requests, pre-trained models of all major categories are now released [here](https://drive.google.com/drive/folders/1nh6SjCGfhiPHymCn_hSkBFU_wfYuMtTm?usp=sharing). Each model is trained to 100 steps with hyper-parameters and performance corresponding to the benchmark table below (using the 6 layers configuration for Transformer-based architectures).

This directory stores the final checkpoint for each model in a zip file, paired with its training log. To use one, unzip the appropriate file, set the config to the same model-type, and run the code with the `-m` flag pointing to the unzipped directory and the `-l` flag pointing to the log file specifically. Note that when using this in "test" mode, you should set `best_model=False` on [this line](https://github.com/VHellendoorn/ICLR20-Great/blob/master/running/run_model.py#L37), since the model at step 100 may not be quite the best model as per the log (though it is consistently close, see the figure below).

These files are hosted on Google Drive due to their size; please create an issue if the link breaks or files are missing. I may switch them to git large file storage at a later date.

## Quick Start
- Tensorflow (2.2.x+)
- python 3.6+
- `pip install -r requirements.txt`.

1. clone data using `--depth=1`: [data repository](https://github.com/google-research-datasets/great)
2. TRAIN: `python running/run_model.py *data_dir* vocab.txt config.yml`
3. TEST: `python running/run_model.py *data_dir* vocab.txt config.yml -m *model_path* -l *log_path* -e True`
