import python
import external.CodeDuplication

from Copy id,string relativePath,int equivClass
where duplicateCode(id,relativePath,equivClass)
select id,
       relativePath.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       equivClass