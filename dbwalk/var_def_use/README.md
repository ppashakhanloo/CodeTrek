# DBwalk for variable define-use classification. 


## offline setting

In this setting, we train an test using offline sampled random walks from the db graphs. 

### Data preparation

The raw data can be downloaded from `:drive/dataset/defuse_offline.tar.gz`. After decompress, you should find

```
defuse_offline/
|___train/
|   |__walks_train_file_xxx.json
|___dev/
|___eval/
|___all_labels.txt
|___other csv files that summarizes the stats
```

Then run `run_cook_data.sh`. Make sure to adapt the data_root paramter to your own setup. 
After that you should additionally see the following things under the data folder:
```
defuse_offline/
|___cooked_train/
|   |__chunk_xx.pkl
|___cooked_dev/
|___cooked_eval/
|___dict.pkl
|___other files from raw data
```

### Training

Run `run_main.sh`, make sure to edit the data path to your own setting.
You should be able to see ~97 AUC after 10mins of training.
