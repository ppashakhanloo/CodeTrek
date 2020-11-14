#!/usr/bin/env python

#This file needs to be able to handle all versions of Python we are likely to encounter
#Which is probably 2.6 and upwards

'''Run index.py in buildtools'''

from __future__ import print_function, division

import os
import sys
from python_tracer import getzipfilename

if not "SEMMLE_DIST" in os.environ:
    os.environ["SEMMLE_DIST"] = os.environ["CODEQL_EXTRACTOR_PYTHON_ROOT"]

tools = os.path.join(os.environ['SEMMLE_DIST'], "tools")
zippath = os.path.join(tools, getzipfilename())
sys.path = [ zippath ] + sys.path

import buildtools.index
buildtools.index.main()
