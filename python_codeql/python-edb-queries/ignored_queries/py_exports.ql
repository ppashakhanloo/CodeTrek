import python

from Module id,string name
where py_exports(id,name)
select id,
       name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")