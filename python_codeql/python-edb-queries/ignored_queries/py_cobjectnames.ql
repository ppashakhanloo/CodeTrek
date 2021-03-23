import python

from Object obj,string name
where py_cobjectnames(obj,name)
select obj,
       name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")