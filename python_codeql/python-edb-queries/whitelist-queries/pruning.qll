import python

predicate isSourceLocation(Location loc) {
    loc.getFile().toString().matches("%/datasets/cubert/py_files/%")
}

predicate isLibraryLocation(Location loc) {
    not isSourceLocation(loc)
}
