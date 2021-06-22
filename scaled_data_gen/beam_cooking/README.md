## How to generate a dict.pkl for a set of walks (offline) or graphs (online):
```
python cook_dict_offline.py varmisuse-defuse us-east1 all_vm_offline.txt vm_labels.txt
# or
python cook_dict_online.py varmisuse-defuse us-east1 all_vm_online.txt vm_labels.txt

```

`all_vm_offline.txt` contains the paths to stub files relative to `varmisuse-defuse` bucket.
`vm_labels.txt` is the same as `all_labels.txt`.
