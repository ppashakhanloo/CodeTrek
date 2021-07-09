
##EXCEPTION CLASSIFICATION

```
python exception_cls/run_classifier.py TRAIN test_data/exception_cls vocab.txt bert_config.json saved_model
python exception_cls/run_classifier.py EVAL test_data/exception_cls saved_model saved_model/model.ckpt-<num>

```
