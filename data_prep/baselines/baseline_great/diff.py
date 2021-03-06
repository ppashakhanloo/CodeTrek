import re
import os
import sys

def findnth(string, substring, n):
  parts = list(re.split('(?<=[,|\.|=|\(|\)|\[|\]| +|:|-|\+|\*|\\|/|<|>|!=|\{|\}])' + substring + '(?=,|\.|=|\(|\)|\[|\]| +|:|-|\+|\*|\\|/|<|>|!=|\{|\})', string, maxsplit=n+1))
  if len(parts) <= n + 1:
    return -1
  return len(string) - len(parts[-1]) - len(substring)

def get_diff(file1, file2):

  with open(file1, 'r') as f1:
    with open(file2, 'r') as f2:
      file1_lines = f1.readlines()
      file2_lines = f2.readlines()
      row, col = 0, 0
      for line_number in range(len(file1_lines)):
        if file1_lines[line_number] != file2_lines[line_number]:
          # diff line found, let's find the token
          row = line_number
          break
      # here, we have the row.
      if row != 0:
        for char_number in range(len(file1_lines[row])):
          if file1_lines[row][char_number] != file2_lines[row][char_number]:
            col = char_number
            break
    
      line1 = re.split(',|\.|=|\(|\)|\[|\]| +|:|-|\+|\*|\\|/|<|>|!=|\{|\}', file1_lines[row].strip())
      line2 = re.split(',|\.|=|\(|\)|\[|\]| +|:|-|\+|\*|\\|/|<|>|!=|\{|\}', file2_lines[row].strip())

      for index in range(len(line1)):
        if line1[index] != line2[index]:
          # taking a different approach since splitting causes a ton of headaches
          num_tok_comm = 0
          tok1 = line1[index]
          tok2 = line2[index]
          idx = 0
          while idx < min(len(tok1), len(tok2)) and tok1[idx] == tok2[idx]:
            idx += 1
        
          # shift col behind by idx
          col -= idx
          
          return os.path.abspath(file1), line1[index], row+1, col+1, row+1, col+len(line1[index])+1, os.path.abspath(file2), line2[index], row+1, col+1, row+1, col+len(line2[index])+1
          assert tok1 == file1_lines[row][col:col+len(tok1)] and tok2 == file2_lines[row][col:col+len(tok2)]

          break
  return None
