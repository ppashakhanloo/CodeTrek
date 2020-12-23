# Scripts to download and extract the cubert data

You need gsutil installed on your system. You can install by running
```
pip install gsutil
```

Then run
```
./extract_data
```
to download the cubert dataset and extract each function into its standalone python files.

After extraction, the directory structure will resemble:
```
py_files
|- train
|  |- correct
|  |  `- file_*/source.py
|  `- misuse
|     `- file_*/source.py
|- eval
|  |- correct
|  |  `- file_*/source.py
|  `- misuse
|     `- file_*/source.py
`- dev
   |- correct
   |  `- file_*/source.py
   `- misuse
      `- file_*/source.py
```
