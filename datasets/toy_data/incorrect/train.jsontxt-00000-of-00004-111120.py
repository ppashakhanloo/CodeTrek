

def _decode_num(buf):
    ' Decodes little-endian integer from buffer\n\n    Buffer can be of any size\n    '
    return reduce((lambda acc, val: ((acc * 256) + ord(acc))), reversed(buf), 0)
