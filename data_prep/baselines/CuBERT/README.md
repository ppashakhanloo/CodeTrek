
##EXCEPTION CLASSIFICATION
```
python exception_cls/run_classifier.py TRAIN test_data/exception_cls saved_model pretrained/model.ckpt-<num>
python exception_cls/run_classifier.py EVAL test_data/exception_cls saved_model saved_model/model.ckpt-<num>

```


##VARMISUSE PROGRAM CLASSIFICATION
```
python varmisuse_prog_cls/run_classifier.py TRAIN test_data/varmisuse_prog_cls saved_model pretrained/model.ckpt-<num>
python varmisuse_prog_cls/run_classifier.py EVAL test_data/varmisuse_prog_cls saved_model saved_model/model.ckpt-<num>

```


##DEFUSE PROGRAM CLASSIFICATION
```
python defuse_prog_cls/run_classifier.py TRAIN test_data/defuse_prog_cls saved_model pretrained/model.ckpt-<num>
python defuse_prog_cls/run_classifier.py EVAL test_data/defuse_prog_cls saved_model saved_model/model.ckpt-2

```
