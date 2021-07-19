
def read_csv(filepath):
  items = []
  with open(filepath, 'r') as f:
    for row in f.readlines()[1:]:
      items.append(row.strip().split(','))
  return items

def read_lines(filepath):
  items = []
  with open(filepath, 'r') as f:
    for line in f.readlines():
      items.append(line.strip())
  return items

def log(filepath, content):
  with open(filepath, "a") as f:
    f.write(content + "\n")
