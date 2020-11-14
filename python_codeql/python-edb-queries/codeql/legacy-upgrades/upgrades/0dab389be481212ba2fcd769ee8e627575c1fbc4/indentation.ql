class File extends @file {
  string toString() { result = "file" }
}

from File f, int lineno, string c, int d
where
  exists (@line line, string src, @location loc |
    lines(line, _, src, _) and
    hasLocation(line, loc) and
    locations_default(loc, f, lineno, _, _, _) and
    c = src.regexpCapture("((\\s)\\2*)\\S.*", 2) and
    d = src.regexpCapture("((\\s)\\2*)\\S.*", 1).length()
  )
select f, lineno, c, d