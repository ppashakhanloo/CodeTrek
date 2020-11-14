#!/usr/bin/env python3


import subprocess
import sys, os
import re

def getenv_mandatory(name):
    value = os.environ.get(name)
    if not value:
        print("Error: environment var", name, "not set.")
        sys.exit(1)
    return value

LGTM_SRC = getenv_mandatory('LGTM_SRC')
SOURCE_ARCHIVE = getenv_mandatory('CODEQL_EXTRACTOR_CPP_SOURCE_ARCHIVE_DIR')
TRAP_FOLDER = getenv_mandatory('CODEQL_EXTRACTOR_CPP_TRAP_DIR')

def trap_escape(s):
    return re.sub('"', '""', s)

# Returns an array of absolute file names denoting headers that were used
# during compilation but are outside the project source tree.
def get_used_system_headers():
    lgtm_src = os.path.realpath(LGTM_SRC)
    headers = []
    for root, _, files in os.walk(SOURCE_ARCHIVE):
        base_path = os.path.join('/', os.path.relpath(root, SOURCE_ARCHIVE))
        for f in files:
            f_abs = os.path.join(base_path, f)
            # Exclude files within the source tree
            if not f_abs.startswith(lgtm_src + os.sep):
                headers.append(f_abs)
    return headers

# Returns the "upstream version" portion of a given Debian package version.
def dpkg_upstream_version(v):
    # The manpage deb-version(5) claims to specify how to separate the upstream
    # version from the distribution-specific component, but it does not account
    # for all the strange conventions we see on Ubuntu.
    v = re.sub('^\d+:', '', v)
    v = re.sub('-[^-]+$', '', v)
    v = re.sub('ubuntu.*$', '', v)
    v = re.sub('\.dfsg.*$', '', v)
    v = re.sub('\+.+$', '', v)
    v = re.sub('~.+$', '', v)
    return v

# Returns a dict from header file names to names of Debian packages installed
# on the system. This is done by calling `dpkg-query`. The returned package
# names may contain architecture tags for disambiguation.
def dpkg_headers_to_bin_packages():
    try:
        process = subprocess.Popen(
                ['dpkg-query', '-S', '*.h*', '-S', '/usr/include/*'],
                bufsize=8192,
                stdout=subprocess.PIPE, universal_newlines=True)
    except OSError:
        print("Could not execute 'dpkg-query'. Continuing.")
        return {}

    d = {}
    for line in process.stdout:
        if re.search(', .*: ', line):
            # Path is in multiple packages, so it's a directory. Skip it.
            continue
        matchobj = re.match(r'^([^ ]+): (.+)\n$', line)
        if not matchobj:
            print("Warning: cannot parse line from dpkg-query: '" + line + "'")
            continue
        d[matchobj.group(2)] = matchobj.group(1)

    status_code = process.wait()
    if status_code != 0:
        print("Error: dpkg-query exited with status code", status_code)
        sys.exit(0)

    return d

# Returns a dict describing all packages known to `dpkg-query`. Each key is the
# name of a binary package, possibly including an architecture tag for
# disambiguation. Each value is a dict with keys 'source_package',
# 'upstream_version', and 'trap_id'.
def dpkg_bin_to_source_info():
    try:
        # - The ${binary:Package} format, unlike ${Package}, appends an
        #   architecture tag where necessary so it matches the output of
        #   "dpkg-query --search".
        # - The ${source:Package} format, unlike ${Source}, is always defined
        #   and never includes version information.
        # - The ${source:Version} format appears to be slightly closer to the
        #   upstream version than ${Version} is.
        process = subprocess.Popen(
                ['dpkg-query', '--show',
                 '--showformat=${binary:Package}\t'+
                              '${source:Package}\t'+
                              '${source:Version}\n'],
                bufsize=8192,
                stdout=subprocess.PIPE,
                universal_newlines=True)
    except OSError:
        print("Could not execute 'dpkg-query'. Continuing.")
        return {}

    d = {}
    for line in process.stdout:
        matchobj = re.match(r'^([^\t]+)\t([^\t]+)\t([^\t]+)\n$', line)
        if not matchobj:
            print("Warning: cannot parse line from dpkg-query: '" + line + "'")
            continue
        info = {
                'source_package': matchobj.group(2),
                'upstream_version': dpkg_upstream_version(matchobj.group(3)),
        }
        info['trap_id'] = info['source_package'] +" "+ info['upstream_version']
        d[matchobj.group(1)] = info

    status_code = process.wait()
    if status_code != 0:
        print("Error: dpkg-query exited with status code", status_code)
        sys.exit(0)

    return d

# Serialize `headers_to_bin` and `bin_to_source_info` in Semmle _trap_ format
# into the file object `output`.
def dpkg_emit_trap(output, headers_to_bin, bin_to_source_info):
    # To avoid defining the same package twice
    packages_emitted = set()

    for header in get_used_system_headers():
        bin_package = headers_to_bin.get(header)
        if not bin_package:
            # Resolve symlinks and see if we can then find it
            bin_package = headers_to_bin.get(os.path.realpath(header))
        if not bin_package:
            # This may happen if the file was not installed through a package
            # manager we support or if it was created by a post-install script.
            continue
        source_info = bin_to_source_info.get(bin_package)
        if not source_info:
            print("Warning: No source package for", bin_package)
            continue
        trap_id = source_info['trap_id']

        if not trap_id in packages_emitted:
            packages_emitted.add(trap_id)
            output.write('external_packages(@"')
            output.write(trap_escape(trap_id))
            output.write('", "dpkg", "')
            output.write(trap_escape(source_info['source_package']))
            output.write('", "')
            output.write(trap_escape(source_info['upstream_version']))
            output.write('")\n')

        output.write('header_to_external_package(@"')
        output.write(trap_escape(header))
        output.write(';sourcefile", @"')
        output.write(trap_escape(trap_id))
        output.write('")\n')


if __name__ == '__main__':
    print(sys.argv[0] + ": starting.")

    if not os.path.isdir(TRAP_FOLDER):
        print(sys.argv[0] + ": no trap directory.")
        sys.exit(0)

    headers_to_bin = dpkg_headers_to_bin_packages()
    bin_to_source_info = dpkg_bin_to_source_info()

    output_file_name = os.path.join(TRAP_FOLDER, 'external_packages.trap')
    with open(output_file_name, 'w') as trap:
        dpkg_emit_trap(trap, headers_to_bin, bin_to_source_info)

    print(sys.argv[0] + ": done.")
