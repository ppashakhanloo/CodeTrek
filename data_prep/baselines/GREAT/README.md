## Quick Start
- Tensorflow (2.2.x+)
- python 3.6+
- `pip install -r requirements.txt`.

# VARMISUSE LOCALIZATION AND REPAIR
1. TRAIN: `python varmisuse_loc_rep/run_model.py test_data/varmisuse_loc_rep vocab.txt config.yml -m saved_model -l saved_model/log.txt`
2. TEST: `python varmisuse_loc_rep/run_model.py test_data/varmisuse_loc_rep vocab.txt config.yml -m saved_model -l saved_model/log.txt -e True`

# DEFUSE PROGRAM CLASSIFICATION
1. TRAIN: `python varmisuse_prog_cls/run_model.py test_data/varmisuse_prog_cls vocab.txt config.yml -m saved_model -l saved_model/log.txt`
2. TEST: `python varmisuse_prog_cls/run_model.py test_data/varmisuse_prog_cls vocab.txt config.yml -m saved_model -l saved_model/log.txt -e True`

# DEFUSE PROGRAM CLASSIFICATION
1. TRAIN: `python defuse_prog_cls/run_model.py test_data/defuse_prog_cls vocab.txt config.yml -m saved_model -l saved_model/log.txt`
2. TEST: `python defuse_prog_cls/run_model.py test_data/defuse_prog_cls vocab.txt config.yml -m saved_model -l saved_model/log.txt -e True`

# EXCEPTION CLASSIFICATION
1. TRAIN: `python exception_cls/run_model.py test_data/exception_cls vocab.txt config.yml -m saved_model -l saved_model/log.txt`
2. TEST: `python exception_cls/run_model.py test_data/exception_cls vocab.txt config.yml -m saved_model -l saved_model/log.txt -e True`
