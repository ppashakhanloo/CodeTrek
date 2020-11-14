#!/usr/bin/python
import errno
import os
import re
from subprocess import *
import sys

if (len(sys.argv) != 5):
    # This should never happen, so we allow ourselves the luxury of
    # being noisy here.
    sys.exit('Bad args: ' + str(sys.argv))

try:
  unsign = sys.argv[1]
  # The file we're copying from - may be a symlink.
  orig_file = sys.argv[2]
  binary = sys.argv[3]
  dest = sys.argv[4]


  # Dereference all symlinks.
  real_file = os.path.realpath(orig_file)
  real_location = os.path.dirname(real_file)

  # Work out where we really want to put the relocated executable
  # `dest` is the path the user will invoke, but may need to be a
  # symlink to `real_dest` - the actual executable.
  # `dest` has the form `<PREFIX> + orig_file`
  # `real_dest` needs to have the form `<PREFIX> + real_file`.
  real_dest = dest[0:len(dest) - len(orig_file)] + real_file

  otool = Popen(['otool', '-l', binary], stdout=PIPE, stderr=PIPE)
  output, errors = otool.communicate()
  # We consume stderr to stop it polluting our stderr, but we then just
  # ignore it.

  rpath_re = re.compile(r'         path @(executable_path|loader_path)(.*) \(offset [0-9]*\)')
  lib_re = re.compile(r'         name @(executable_path|loader_path)(.*) \(offset [0-9]*\)')
  for line in output.split('\n'):
      m = rpath_re.match(line)
      if m:
          if call(['install_name_tool',
                       '-rpath', '@' + m.group(1) + m.group(2),
                                 real_location + m.group(2),
                       binary]) != 0:
              # We have to fail silently, as we can't pollute stdout/stderr
              # during a build.
              sys.exit(1)
      m = lib_re.match(line)
      if m:
          if call(['install_name_tool',
                       '-change', '@' + m.group(1) + m.group(2),
                                  real_location + m.group(2),
                       binary]) != 0:
              # We have to fail silently, as we can't pollute stdout/stderr
              # during a build.
              sys.exit(1)

  unsigned_binary = binary + "-unsigned"
  if call([unsign, binary, unsigned_binary]) != 0:
    # We have to fail silently, as we can't pollute stdout/stderr
    # during a build.
    sys.exit(1)

  # If dest and real_dest are different we need to set up a symlink.
  if dest != real_dest:
      try:
          # libtrace will have ensured that dirname(dest) exists but not
          # dirname(real_dest).  So we need to create it:
          real_dest_dir = os.path.dirname(real_dest)
          if not os.path.isdir(real_dest_dir):
              os.makedirs(real_dest_dir)
          os.symlink(real_dest, dest)
      except OSError, e:
          # Swallow not being able to create the symlink if it already points
          # to the correct destination.
          if e.errno == errno.EEXIST:
              if os.path.realpath(dest) != real_dest:
                  raise

  # And put the binary in place
  os.rename(unsigned_binary, real_dest)

except:
  # We have to fail silently, as we can't pollute stdout/stderr
  # during a build.
  sys.exit(1)

