from utils import tokenizer_registry

def tokenize(string, language):
  if language == 'python':
    tokenizer = tokenizer_registry.TokenizerEnum.PYTHON.value()
  elif language == 'java':
    tokenizer = tokenizer_registry.TokenizerEnum.JAVA.value()
  else:
    raise NotImplementedError
  
  return tokenizer.tokenize(string)[:-1]
