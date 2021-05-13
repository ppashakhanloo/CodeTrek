import python

predicate isSourceLocation(Location loc) {
    loc.getFile().toString().matches("%/tmp/%")
}

predicate isLibraryLocation(Location loc) {
    not isSourceLocation(loc)
}
