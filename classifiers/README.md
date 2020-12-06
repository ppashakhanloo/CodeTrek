Example usage:

1. SVM Classifier for variable misuse
```
python3 ../classifiers/SVM_classifier.py edges/correct edges/incorrect
```
In `SVM_classifier.py` you can specify the number of `TRAIN` and `TEST` data points as well as `EDGE_LIMIT`.

The structure of the edges directory:
```
edges
|__ correct
|   |__ edgefile1...
|
|__ incorrect
|   |__ edgefile1...
|
```

2. SVM Multiclass Classifier for exception classification
```
python3 ../classifiers/SVM_multi_classifier.py edges
```
The structure of `edges` directory:
```
edges
|__ train
|   |__ labels.txt
|   |__ edgefile1...
|   |__ ...
|__ test
|   |__ labels.txt
|   |__ edgefile1...
|   |__ ...
|
```
