import python
import external.VCS

from Commit id,string revision,string author,date revisionDate,int changeSize
where svnentries(id,revision,author,revisionDate,changeSize)
select id,
       revision.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       author.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       revisionDate,
       changeSize