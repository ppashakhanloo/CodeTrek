import python

from Module module_,string version
where py_extracted_version(module_,version)
select module_,
       version.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")