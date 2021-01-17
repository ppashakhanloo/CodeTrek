import java

predicate isSourceLocation(Location loc) {
    loc.getFile().getAbsolutePath().matches("%/datasets/%")
}

predicate isLibraryLocation(Location loc) {
    not isSourceLocation(loc)
}
