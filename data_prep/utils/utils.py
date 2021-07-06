
def log(filepath, content):
  with open(filepath, "a") as f:
    f.write(content + "\n")
