import python

from Container file,string contents
where file_contents(file,contents)
select file,
       contents.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")