
string version() {
    py_flags("version.major", result)
}

class CObject extends @py_cobject { string toString() { none() } }

from CObject object, string name, CObject member
where py_cmembers(object,name,member)
select object, name, member, version()
