import python
import external.CodeDuplication

from Copy id,string relativePath,int equivClass
where similarCode(id,relativePath,equivClass)
select id,
       relativePath.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       equivClass