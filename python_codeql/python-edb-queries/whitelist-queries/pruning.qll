import python

predicate isSourceLocation(Location loc) {
    loc.getFile().toString().matches("%/datasets/%")
}

predicate isLibraryLocation(Location loc) {
    not isSourceLocation(loc)
}
