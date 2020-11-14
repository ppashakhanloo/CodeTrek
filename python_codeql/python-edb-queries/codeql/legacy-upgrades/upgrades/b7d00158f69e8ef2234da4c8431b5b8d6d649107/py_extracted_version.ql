
string version() {
    py_flags("version.major", result)
}

class Module extends @py_Module { string toString() { none() } }

from Module mod
select mod, version()
