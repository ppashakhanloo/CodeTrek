# CuBERT


## Update 2021/9/22: Evaluating and Training the Models

A `run_classifier.py` script (forked from the original BERT version) is provided
to use the finetuned models for the classification tasks above.

To use it, you first need to download the relevant files above (i.e., the
corresponding vocabulary, dataset, and model checkpoint) and then need to create
a BERT configuration file matching the chosen model.

Assuming the downloaded data are stored in `$DATA_DIR`, you can then use the
following command line to evaluate a model (note that it requires access to the
`bert` module in your python library path):

```
python cubert/run_classifier.py
  --do_train=False
  --bert_config_file=$DATA_DIR/bert_large_config.json
  --vocab_file=$DATA_DIR/github_python_minus_ethpy150open_deduplicated_vocabulary.txt
  --task_name=exception
  --init_checkpoint=$DATA_DIR/exception__epochs_20__pre_trained_epochs_1/model.ckpt-378
  --data_dir=$DATA_DIR/exception_datasets
  --output_dir=exception_results
  --do_eval=True
```

This example file was contributed by Marc Brockschmidt <marc+github@marcbrockschmidt.de>. We are grateful for his help!


## Update 2021/7/11: Fresh Pre-trained Python and Java Models

We are releasing a fresh set of Python and Java pre-training corpus and models, drawn from the BigQuery version of GitHub as of July 11, 2021. These pre-training corpora were deduplicated with the updated process described in [Collection Query](https://github.com/google-research/google-research/tree/master/cubert#collection-query) below. Note that for the Python corpus, files similar to ETH Py150 Open are also extracted from pre-training. The Java corpus is just internally deduplicated.

The pre-trained models were BERT Large, and trained for 2 epochs.

* Python, deduplicated, BigQuery snapshot as of July 11, 2021.
    * Manifest: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20210711_Python/github_python_minus_ethpy150open_deduplicated_manifest)
        [`gs://cubert/20210711_Python/github_python_minus_ethpy150open_deduplicated_manifest`].
    * Vocabulary: [[UI]](https://storage.cloud.google.com/cubert/20210711_Python/github_python_minus_ethpy150open_deduplicated_vocabulary.txt)
        [`gs://cubert/20210711_Python/github_python_minus_ethpy150open_deduplicated_vocabulary.txt`].
    * Model checkpoint for length 512, 2 epochs: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20210711_Python/pre_trained_model_epochs_2__length_512)
        [`gs://cubert/20210711_Python/pre_trained_model_epochs_2__length_512`].
    * Model checkpoint for length 1024, 2 epochs: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20210711_Python/pre_trained_model_epochs_2__length_1024)
        [`gs://cubert/20210711_Python/pre_trained_model_epochs_2__length_1024`].
    * Model checkpoint for length 2048, 2 epochs: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20210711_Python/pre_trained_model_epochs_2__length_2048)
        [`gs://cubert/20210711_Python/pre_trained_model_epochs_2__length_2048`].

* Java, deduplicated, BigQuery snapshot as of July 11, 2021.
    * Manifest: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20210711_Java/github_java_deduplicated_manifest)
        [`gs://cubert/20210711_Java/github_java_deduplicated_manifest`].
    * Vocabulary: [[UI]](https://storage.cloud.google.com/cubert/20210711_Java/github_java_deduplicated_vocabulary.txt)
        [`gs://cubert/20210711_Java/github_java_deduplicated_vocabulary.txt`].
    * Model checkpoint for length 512, 2 epochs: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20210711_Java/pre_trained_model_epochs_2__length_512)
        [`gs://cubert/20210711_Java/pre_trained_model_epochs_2__length_512`].
    * Model checkpoint for length 1024, 2 epochs: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20210711_Java/pre_trained_model_epochs_2__length_1024)
        [`gs://cubert/20210711_Java/pre_trained_model_epochs_2__length_1024`].
    * Model checkpoint for length 2048, 2 epochs: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20210711_Java/pre_trained_model_epochs_2__length_2048)
        [`gs://cubert/20210711_Java/pre_trained_model_epochs_2__length_2048`].


## Update 2021/03/04: Clarifications

### Errata in Pre-training Corpus

The paper described how we constructed the Python pre-training corpus in detail. We discovered a small deviation in the actual process we followed, which although harmless to the validity of the results, may cause confusion. We explain this deviation here.

We collect our pre-training corpus as follows:

1. Read from BigQuery's GitHub database all files ending in `.py` that are not symbolic links and appear in the `master/head` branch.
2. Remove from the results of Step #1 those files that are similar (for some approximate similarity metric) to the files in the ETH Py150 Open corpus.
3. Keep from the results of Step #2 one file per "similarity" cluster, i.e., groups of files that are similar to each other (according to the same approximate similarity metric used in Step #2).

What happened in practice was the following:

1. Step #1 brought in some duplicate files from GitHub, due to cloning (same content digest but different repository and path).
2. Step #2 removed from the results of Step #1 all files similar to ETH Py150 Open, as intended. That included all clones of such files.
3. Step #3 didn't remove all identical files from its input. The reason for this was that Step #3 is essentially an O(N^2) process, and N was roughly 14M. To speed the process up, we performed Step #3 in semi-independent batches. As a result, if a file had a clone in another batch, they might both be individually kept by their corresponding batches. That problem did not affect Step #2, because all GitHub files were compared to all ETH Py150 Open files. Therefore, Step #2 was not affected.

Out of the ~7M files in our pre-training corpus, only ~4M files are indeed unique. Consequently, the manifest contains multiple entries for some GitHub SHA1 digests. In practice, this causes a small skew in our pre-training process. Note, however, that this does not affect the validity of using ETH Py150 Open as our fine-tuning corpus, or our results, because there is still no "information leak" between the pre-training and fine-tuning corpora.

We are indebted to David Gros (@DNGros) for discovering this in our datasets, and bringing it to our attention.

### Collection Query

We have been asked about the query used to fetch the pre-training corpus from BigQuery's GitHub dataset. We provide it here:

```
with
  allowed_repos as (
    select repo_name, license from `bigquery-public-data.github_repos.licenses`
    where license in unnest(
      ["artistic-2.0", "isc", "mit", "cc0-1.0", "epl-1.0", "gpl-2.0",
       "gpl-3.0", "mpl-2.0", "lgpl-2.1", "lgpl-3.0", "unlicense", "apache-2.0",
       "bsd-2-clause"])),
  github_files_at_head as (
    select id, repo_name, path as filepath, symlink_target
    from `bigquery-public-data.github_repos.files`
    where ref = "refs/heads/master" and ends_with(path, ".py")
    and symlink_target is null),
  unique_full_path AS (
    select id, max(concat(repo_name, ':', filepath)) AS full_path
    from github_files_at_head
    group by id),
  unique_github_files_at_head AS (
    select github_files_at_head.id, github_files_at_head.repo_name,
      github_files_at_head.filepath
    from github_files_at_head, unique_full_path
    where concat(github_files_at_head.repo_name, ':',
                 github_files_at_head.filepath) = unique_full_path.full_path),
  github_provenances as (
    select id, allowed_repos.repo_name as repo_name, license, filepath
    from allowed_repos inner join unique_github_files_at_head
    on allowed_repos.repo_name = unique_github_files_at_head.repo_name),
  github_source_files as (
    select id, content
    from `bigquery-public-data.github_repos.contents`
    where binary = false),
  github_source_snapshot as (
    select github_source_files.id as id, repo_name as repository, license,
      filepath,content
    from github_source_files inner join github_provenances
    on github_source_files.id = github_provenances.id)
select * from github_source_snapshot;
```

Note that this corrects for the error described in the previous subsection.

The original query, used for the pre-training corpus in the paper, did not have
the ID-based deduplication done by views `unique_full_path` and
`unique_github_files_at_head`. Specifically, view `github_provenances` was
reading from `github_files_at_head` in its `from` clause, rather than from
`unique_github_files_at_head`.


## Update 2020/11/16: Pre-trained Java Model with Code Comments

We are releasing a Java pre-training corpus and pre-trained model. This model was pre-trained on all Java content, including comments.

* Java, deduplicated, with code comments, BigQuery snapshot as of October 18, 13, 2020.
    * Manifest: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20201018_Java_Deduplicated/github_java_manifest)
        [`gs://cubert/20201018_Java_Deduplicated/github_java_manifest`].
    * Vocabulary: [[UI]](https://console.cloud.google.com/storage/browser/_details/cubert/20201018_Java_Deduplicated/github_java_vocabulary.txt)
        [`gs://cubert/20201018_Java_Deduplicated/github_java_vocabulary.txt`].
    * Model checkpoint for length 1024, 1 epoch: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20201018_Java_Deduplicated/pre_trained_model_deduplicated__epochs_1__length_1024)
        [`gs://cubert/20201018_Java_Deduplicated/pre_trained_model_deduplicated__epochs_1__length_1024`].


## Update 2020/09/29: Pre-trained Java Model

We are releasing a Java pre-training corpus and pre-trained model. This model was not pre-trained on comments, but an expanded model including Javadoc and regular comments is upcoming.

* Java, deduplicated, no code comments, BigQuery snapshot as of September 13, 2020.
    * Manifest: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200913_Java_Deduplicated/github_java_manifest)
        [`gs://cubert/20200913_Java_Deduplicated/github_java_manifest`].
    * Vocabulary: [[UI]](https://console.cloud.google.com/storage/browser/_details/cubert/20200913_Java_Deduplicated/github_java_vocabulary.txt)
        [`gs://cubert/20200913_Java_Deduplicated/github_java_vocabulary.txt`].
    * Model checkpoint for length 1024, 1 epoch: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200913_Java_Deduplicated/pre_trained_model_deduplicated__epochs_1__length_1024)
        [`gs://cubert/20200913_Java_Deduplicated/pre_trained_model_deduplicated__epochs_1__length_1024`].


## Introduction

This is a repository for code, models and data accompanying the ICML 2020 paper
[Learning and Evaluating Contextual Embedding of Source Code](http://proceedings.mlr.press/v119/kanade20a.html). In addition to the Python artifacts described in the paper, we are also
releasing the pre-training corpus and CuBERT models for other languages.

If you use the code, models or data released through this repository, please
cite the following paper:
```
@inproceedings{cubert,
author    = {Aditya Kanade and
             Petros Maniatis and
             Gogul Balakrishnan and
             Kensen Shi},
title     = {Learning and evaluating contextual embedding of source code},
booktitle = {Proceedings of the 37th International Conference on Machine Learning,
               {ICML} 2020, 12-18 July 2020},
series    = {Proceedings of Machine Learning Research},
publisher = {{PMLR}},
year      = {2020},
}
```

## The CuBERT Tokenizer

The CuBERT tokenizer for Python is implemented in `python_tokenizer.py`, as
a subclass of a language-agnostic tokenization framework in
`cubert_tokenizer.py`. `unified_tokenizer.py` contains useful utilities for
language-agnostic tokenization,
which can be extended along the lines of the Python tokenizer for other
languages. We show one other instance, for Java, in `java_tokenizer.py`,
although the original CuBERT benchmark is only about Python code.

The code within the `code_to_subtokenized_sentences.py` script can be used for
converting Python code (in fact, any language for which there's a subclass of
`CuBertTokenizer`) into CuBERT sentences. This script can be evaluated on
the `source_code.py.test` file along with a CuBERT subword vocabulary. It should
produce output similar to that illustrated in the
`subtokenized_source_code.py.json` file. To obtain token-ID sequences for use
with TensorFlow models, the `decode_list` logic from
`code_to_subtokenized_sentences.py` can be skipped.

It is possible to configure CuBERT tokenizers to skip emitting tokens of some
kinds. For our fine-tuning tasks presented below, we skip comment and whitespace
tokens. After initializing a tokenizer, this will configure it to skip
those kinds of tokens:
```
from cubert import unified_tokenizer
from cubert import python_tokenizer
...
tokenizer = python_tokenizer.PythonTokenizer()
tokenizer.update_types_to_skip((
      unified_tokenizer.TokenKind.COMMENT,
      unified_tokenizer.TokenKind.WHITESPACE,
  ))
```

## The Multi-Headed Pointer Model

The `finetune_varmisuse_pointer_lib.py` file provides an implementation of the
multi-headed pointer model described in [Neural Program Repair by Jointly Learning to Localize and Repair](https://openreview.net/pdf?id=ByloJ20qtm) on top of the pre-trained CuBERT
model. The `model_fn_builder` function should be integrated into an appropriate
fine-tuning script along the lines of the [fine-tuning script of the BERT model](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_classifier.py#L847).

## Pre-trained Models and Pre-training Corpora

We provide the following files, all stored in Google Cloud Storage. We give
links to each file or directory (via the Cloud Storage UI), as well as URIs for the
corresponding dataset to be used via the [command-line interface](https://cloud.google.com/storage/docs/gsutil).

For each language we provide the following:

1. Manifest of files used during pre-training. This contains the precise specification of pre-training source files, which can be used with BigQuery or the GitHub API (see below for a sample query for BigQuery). The data are stored as sharded text files. Each text line contains a JSON-formatted object.
      * `repository`: string, the name of a GitHub repository.
      * `filepath`: string, the path from the repository root to the file mentioned.
      * `license`: string, (one of 'apache-2.0', 'lgpl-2.1', 'epl-1.0', 'isc', 'bsd-3-clause', 'bsd-2-clause', 'mit', 'gpl-2.0', 'cc0-1.0', 'lgpl-3.0', 'mpl-2.0', 'unlicense', 'gpl-3.0'); the license under which the file’s repository was released on GitHub.
      * `id`: string, a unique identifier under which the file’s content is hosted in BigQuery’s public GitHub repository.
      * `url`: string, a URL by which the GitHub API uniquely identifies the content.

1. Vocabulary file.
Used to encode pre-training and fine-tuning examples. It is extracted from the files pointed to by the Manifest of files. It is stored as a single text file, holding one quoted token per line, as per the format produced by [`tensor2tensor`'s `SubwordTextEncoder`](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py).

1. Pre-trained model checkpoint. These are as stored by BERT's [`run_pretraining.py`](https://github.com/google-research/bert/blob/master/run_pretraining.py). Although we used a modified version of the pre-training code to use the CuBERT tokenizer (see above), the models still have the BERT architecture and are stored in a compatible way. The actual BERT configuration is BERT-Large:
      * "attention_probs_dropout_prob": 0.1,
      * "hidden_act": "gelu",
      * "hidden_dropout_prob": 0.1,
      * "hidden_size": 1024,
      * "initializer_range": 0.02,
      * "intermediate_size": 4096,
      * "num_attention_heads": 16,
      * "num_hidden_layers": 24,
      * "type_vocab_size": 2,
      * "vocab_size": *corresponding vocabulary size*,
      * "max_position_embeddings": *corresponding sequence length*

To retrieve a pre-training file, given its `id`, you can use the following [BigQuery query](https://console.cloud.google.com/bigquery):
```
select files.repo_name, files.path, files.ref, contents.content
from `bigquery-public-data.github_repos.files` as files,
     `bigquery-public-data.github_repos.contents` as contents
where contents.id = files.id and
      contents.id = <id>;
```

At this time, we release the following pre-trained model and pre-training corpus. Look in the updates, below, for other releases.

* Python, deduplicated after files similar to [ETH Py150 Open](https://github.com/google-research-datasets/eth_py150_open) were removed. BigQuery snapshot as of June 21, 2020. These are the models and manifests involved in the published paper.
    * Manifest: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_manifest)
        [`gs://cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_manifest`].
    * Vocabulary: [[UI]](https://console.cloud.google.com/storage/browser/_details/cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_vocabulary.txt)
        [`gs://cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_vocabulary.txt`].
    * Model checkpoint for length 512, 1 epoch: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/pre_trained_model__epochs_1__length_512)
        [`gs://cubert/20200621_Python/pre_trained_model__epochs_1__length_512`].
    * Model checkpoint for length 512, 2 epochs: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/pre_trained_model__epochs_2__length_512)
        [`gs://cubert/20200621_Python/pre_trained_model__epochs_2__length_512`].


## Benchmarks and Fine-Tuned Models

Here we describe the 6 Python benchmarks we created. All 6 benchmarks were derived from [ETH Py150 Open](https://github.com/google-research-datasets/eth_py150_open). All examples are stored as sharded text files. Each text line corresponds to a separate example encoded as a JSON object. For each dataset, we release separate training/validation/testing splits along the same boundaries that ETH Py150 Open splits its files to the corresponding splits. The fine-tuned models are the checkpoints of each model with the highest validation accuracy.

1. **Function-docstring classification**. Combinations of functions with their correct or incorrect documentation string, used to train a classifier that can tell which pairs go together. The JSON fields are:
     * `function`: string, the source code of a function as text
     * `docstring`: string, the documentation string for that function. Note that the string is unquoted. To be able to properly tokenize it with the CuBERT tokenizers, you have to wrap it in quotes first. For example, in Python, use `string_to_tokenize = f'"""{docstring}"""'`.
     * `label`: string, one of (“Incorrect”, “Correct”), the label of the example.
     * `info`: string, an unformatted description of how the example was constructed, including the source dataset (always “ETHPy150Open”), the repository and filepath, the function name and, for “Incorrect” examples, the function whose docstring was substituted.
1. **Exception classification**. Combinations of functions where one exception type has been masked, along with a label indicating the masked exception type. The JSON fields are:
     * `function`: string, the source code of a function as text, in which one exception type has been replaced with the special token “__HOLE__”
     * `label`: string, one of (`ValueError`, `KeyError`, `AttributeError`, `TypeError`, `OSError`, `IOError`, `ImportError`, `IndexError`, `DoesNotExist`, `KeyboardInterrupt`, `StopIteration`, `AssertionError`, `SystemExit`, `RuntimeError`, `HTTPError`, `UnicodeDecodeError`, `NotImplementedError`, `ValidationError`, `ObjectDoesNotExist`, `NameError`, `None`), the masked exception type. Note that `None` never occurs in the data and will be removed in a future release.
     * `info`: string, an unformatted description of how the example was constructed, including the source dataset (always “ETHPy150Open”), the repository and filepath, and the fully-qualified function name.
1. **Variable-misuse classification**. Combinations of functions where one use of a variable may have been replaced with another variable defined in the same context, along with a label indicating if this bug-injection has occurred. The JSON fields are:
     * `function`: string, the source code of a function as text.
     * `label`: string, one of (“Correct”, “Variable misuse”) indicating if this is a buggy or bug-free example.
     * `info`: string, an unformatted description of how the example was constructed, including the source dataset (always “ETHPy150Open”), the repository and filepath, the function, and whether the example is bugfree (marked “original”) or the variable substitution that has occurred (e.g., “correct_variable” → “incorrect_variable”).
1. **Swapped-operand classification**. Combinations of functions where one use binary operator’s arguments have been swapped, to create a buggy example, or left undisturbed, along with a label indicating if this bug-injection has occurred. The JSON fields are:
     * `function`: string, the source code of a function as text.
     * `label`: string, one of (“Correct”, “Swapped operands”) indicating if this is a buggy or bug-free example.
     * `info`: string, an unformatted description of how the example was constructed, including the source dataset (always “ETHPy150Open”), the repository and filepath, the function, and whether the example is bugfree (marked “original”) or the operand swap has occurred (e.g., “swapped operands of `not in`”).
1. **Wrong-binary-operator classification**. Combinations of functions where one binary operator has been swapped with another, to create a buggy example, or left undisturbed, along with a label indicating if this bug-injection has occurred. The JSON fields are:
     * `function`: string, the source code of a function as text.
     * `label`: string, one of (“Correct”, “Wrong binary operator”) indicating if this is a buggy or bug-free example.
     * `info`: string, an unformatted description of how the example was constructed, including the source dataset (always “ETHPy150Open”), the repository and filepath, the function, and whether the example is bugfree (marked “original”) or the operator replacement has occurred (e.g., “`==`-> `!=`”).
1. **Variable-misuse localization and repair**. Combinations of functions where one use of a variable may have been replaced with another variable defined in the same context, along with information that can be used to localize and repair the bug, as well as the location of the bug if such a bug exists. The JSON fields are:
     * `function`: a list of strings, the source code of a function, tokenized with the vocabulary from item b. Note that, unlike other task datasets, this dataset gives a tokenized function, rather than the code as a single string.
     * `target_mask`: a list of integers (0 or 1). If the integer at some position is 1, then the token at the corresponding position of the function token list is a correct repair for the introduced bug. If a variable has been split into multiple tokens, only the first subtoken is marked in this mask. If the example is bug-free, all integers are 0.
     * `error_location_mask`: a list of integers (0 or 1). If the integer at some position is 1, then there is a variable-misuse bug at the corresponding location of the tokenized function. In a bug-free example, the first integer is 1. There is exactly one integer set to 1 for all examples. If a variable has been split into multiple tokens, only the first subtoken is marked in this mask.
     * `candidate_mask`: a list of integers (0 or 1). If the integer at some position is 1, then the variable starting at that position in the tokenized function is a candidate to consider when repairing a bug. Candidates are all variables defined in the function parameters or via variable declarations in the function. If a variable has been split into multiple tokens, only the first subtoken is marked in this mask, for each candidate.
     * `provenance`: string, an unformatted description of how the example was constructed, including the source dataset (always “ETHPy150Open”), the repository and filepath, the function, and whether the example is bugfree (marked “original”) or the buggy/repair token positions and variables (e.g., “16/18 `kwargs` → `self`”). 16 is the position of the introduced error, 18 is the location of the repair.


We release the following file collections:

1. **Function-docstring classification**.
    * Dataset: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/function_docstring_datasets)
        [`gs://cubert/20200621_Python/function_docstring_datasets`].
    * Fine-tuned Model: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/function_docstring__epochs_20__pre_trained_epochs_1)
        [`gs://cubert/20200621_Python/function_docstring__epochs_20__pre_trained_epochs_1`].
1. **Exception classification**.
    * Dataset: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/exception_datasets)
        [`gs://cubert/20200621_Python/exception_datasets`].
    * Fine-tuned Model: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/exception__epochs_20__pre_trained_epochs_1)
        [`gs://cubert/20200621_Python/exception__epochs_20__pre_trained_epochs_1`].
1. **Variable-misuse classification**.
    * Dataset: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/variable_misuse_datasets)
        [`gs://cubert/20200621_Python/variable_misuse_datasets`].
    * Fine-tuned Model: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/variable_misuse__epochs_20__pre_trained_epochs_1)
        [`gs://cubert/20200621_Python/variable_misuse__epochs_20__pre_trained_epochs_1`].
1. **Swapped-operand classification**.
    * Dataset: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/swapped_operands_datasets)
        [`gs://cubert/20200621_Python/swapped_operands_datasets`].
    * Fine-tuned Model: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/swapped_operands__epochs_20__pre_trained_epochs_1)
        [`gs://cubert/20200621_Python/swapped_operands__epochs_20__pre_trained_epochs_1`].
1. **Wrong-binary-operator classification**.
    * Dataset: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/wrong_binary_operator_datasets)
        [`gs://cubert/20200621_Python/wrong_binary_operator_datasets`].
    * Fine-tuned Model: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/wrong_binary_operator__epochs_20__pre_trained_epochs_1)
        [`gs://cubert/20200621_Python/wrong_binary_operator__epochs_20__pre_trained_epochs_1`].
1. **Variable-misuse localization and repair**.
    * Dataset: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/variable_misuse_repair_datasets)
        [`gs://cubert/20200621_Python/variable_misuse_repair_datasets`].
    * Fine-tuned Model: [[UI]](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/variable_misuse_repair__epochs_20__pre_trained_epochs_1)
        [`gs://cubert/20200621_Python/variable_misuse_repair__epochs_20__pre_trained_epochs_1`].
