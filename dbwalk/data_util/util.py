from collections import namedtuple

RawData = namedtuple('RawData', ['node_idx', 'edge_idx', 'source', 'label'])
RawFile = namedtuple('RawFile', ['gv_file', 'anchor', 'source', 'label'])
