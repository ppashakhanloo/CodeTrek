import array
import csv
import sys

HOME='/Users/pardis/Google Drive/[University]/Sem 7/Research/CodeQL/'

def read_csv(filename):
  print(filename)
  with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = []
    first = True
    for row in reader:
      if first:
        first = False
        continue
      data.append(row)
    for i in range(len(data)):
      for j in range(len(data[0])):
        data[i][j] = int(data[i][j])
    return data


lst = ['ClassObject', 'Location', 'Class',
       'Expression_location', 'Statement_location',
       'Module', 'Expression', 'Statement', 'Variable', 'Function']

num_nodes = 0
for item in lst:
  data = read_csv(HOME+item+'.csv')
  num_nodes += len(data)
  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')


