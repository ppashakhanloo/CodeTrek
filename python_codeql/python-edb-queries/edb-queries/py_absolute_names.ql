import python

from Module module_,string relname,string absname
where py_absolute_names(module_,relname,absname)
select module_,
       relname.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       absname.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")