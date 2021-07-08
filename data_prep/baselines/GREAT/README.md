## Quick Start
- Tensorflow (2.2.x+)
- python 3.6+
- `pip install -r requirements.txt`.

1. TRAIN: `python running/run_model.py test_data/defuse_prog_cls vocab.txt config.yml -m saved_model -l saved_model/log.txt`
2. TEST: `python running/run_model.py test_data/defuse_prog_cls vocab.txt config.yml -m saved_model -l saved_model/log.txt -e True`
