import python

from Object object,string name,Object member,string version
where py_cmembers_versioned(object,name,member,version)
select object,
       name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       member,
       version.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")