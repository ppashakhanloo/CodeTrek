
string version() {
    py_flags("version.major", result)
}

from string name, string value
where py_flags(name, value)
select name, value, version()
