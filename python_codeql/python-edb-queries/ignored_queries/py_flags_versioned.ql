import python

from string name,string value,string version
where py_flags_versioned(name,value,version)
select name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       value.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       version.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")