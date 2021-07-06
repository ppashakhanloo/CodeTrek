# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This modules demonstrates how to convert code to subtokenized sentences."""
import json
import itertools
from typing import List, Sequence, Tuple


from absl import logging
from tensor2tensor.data_generators import text_encoder

import python_tokenizer
import cubert_tokenizer
import unified_tokenizer


def wordpiece_ids_from_wordpiece_tokens(wordpiece_subtokens, subword_tokenizer):
  return tuple(subword_tokenizer._subtoken_string_to_id[w] for w in wordpiece_subtokens)

def next_whole_token(wordpiece_subtokens, initial_tokenizer, subword_tokenizer):
  wordpiece_ids = wordpiece_ids_from_wordpiece_tokens(wordpiece_subtokens, subword_tokenizer)
  full_cubert_subtokens = (subword_tokenizer._subtoken_ids_to_tokens(wordpiece_ids))
  full_cubert_subtokens.append(unified_tokenizer.quote_special(unified_tokenizer.TokenKind.EOS.name))
  full_whole_tokens = initial_tokenizer.untokenize_agnostic(full_cubert_subtokens)

  if len(full_whole_tokens) < 2:
    raise ValueError(f'Whole tokens {full_whole_tokens} ended up '
                     f'undifferentiable in {wordpiece_subtokens}.')

  whole_token = full_whole_tokens[0]
  for end_index in range(1, len(wordpiece_ids) + 1):
    prefix_list = wordpiece_ids[:end_index]
    partial_cubert_subtokens = ( subword_tokenizer._subtoken_ids_to_tokens(prefix_list))
    partial_cubert_subtokens.append(unified_tokenizer.quote_special(unified_tokenizer.TokenKind.EOS.name))
    partial_whole_tokens = initial_tokenizer.untokenize_agnostic(partial_cubert_subtokens)
    if len(partial_whole_tokens) > 1:
      if partial_whole_tokens[0] == whole_token:
        return whole_token, end_index
  raise ValueError('Could not find a whole token in %r' % (wordpiece_subtokens))

def wordpiece_subtokens_to_code(wordpiece_subtokens, initial_tokenizer, subword_tokenizer):
  wordpiece_ids = wordpiece_ids_from_wordpiece_tokens(wordpiece_subtokens, subword_tokenizer)
  return wordpiece_ids_to_code(wordpiece_ids, initial_tokenizer, subword_tokenizer)

def wordpiece_ids_to_code(wordpiece_ids, initial_tokenizer, subword_tokenizer):
  cubert_subtokens = (subword_tokenizer._subtoken_ids_to_tokens(wordpiece_ids))
  cubert_subtokens.append(unified_tokenizer.quote_special(unified_tokenizer.TokenKind.EOS.name))
  return initial_tokenizer.untokenize(cubert_subtokens)

def code_to_cubert_sentences(code, initial_tokenizer, subword_tokenizer):
  tokens = initial_tokenizer.tokenize(code)[:-1]
  groups_by_endtoken = itertools.groupby(
      tokens, key=lambda x: x == unified_tokenizer.NEWLINE)
  raw_sentences: List[List[str]] = []
  for i, (k, v) in enumerate(groups_by_endtoken):
    tokens = list(v)
    if k:
      if i == 0:
        raw_sentences.extend([[]] * len(tokens))
      else:
        if len(tokens) == 1:
          continue
        elif len(tokens) > 1:
          raw_sentences.extend([[]] * (len(tokens) - 1))
        else:
          raise AssertionError('itertools.groupby seems to have returned an '
                               'empty group: %r' % tokens)
    else:
      raw_sentences.append(tokens)
  sentences = [s + [unified_tokenizer.NEWLINE] for s in raw_sentences]
  subtokenized_sentences = []
  for sentence in sentences:
    encoded_tokens = [subword_tokenizer.encode_without_tokenizing(t) for t in sentence]
    flattened_encodings = sum(encoded_tokens, [])
    decoded_tokens = subword_tokenizer.decode_list(flattened_encodings)
    subtokenized_sentences.append(decoded_tokens)
  return subtokenized_sentences

def subtokenize_string(vocab, s):
  py_tokenizer = python_tokenizer.PythonTokenizer()
  subword_tokenizer = text_encoder.SubwordTextEncoder(vocab)
  return code_to_cubert_sentences(code=s,
                                  initial_tokenizer=py_tokenizer,
                                  subword_tokenizer=subword_tokenizer)

def subtokenize_file(vocab, input_filepath):
  py_tokenizer = python_tokenizer.PythonTokenizer()
  subword_tokenizer = text_encoder.SubwordTextEncoder(vocab)
  code = None
  with open(input_filepath, 'r') as input_file:
    code = input_file.read()
  return code_to_cubert_sentences(code=code,
                                  initial_tokenizer=py_tokenizer,
                                  subword_tokenizer=subword_tokenizer)
