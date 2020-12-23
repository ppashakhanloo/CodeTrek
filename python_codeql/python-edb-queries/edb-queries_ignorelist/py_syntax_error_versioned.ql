import python

from Location id,string message,string version
where py_syntax_error_versioned(id,message,version)
select id,
       message.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       version.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")