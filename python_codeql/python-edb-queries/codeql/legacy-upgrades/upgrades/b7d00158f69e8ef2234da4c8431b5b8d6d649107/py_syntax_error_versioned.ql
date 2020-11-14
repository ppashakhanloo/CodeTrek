
string version() {
    py_flags("version.major", result)
}

class Location extends @location { string toString() {none()} }

from Location loc, string msg
where py_syntax_error(loc, msg)
select loc, msg, version()
