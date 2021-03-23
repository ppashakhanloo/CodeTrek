import python
import external.VCS

from Commit commit,Container file,int addedLines,int deletedLines
where svnchurn(commit,file,addedLines,deletedLines)
select commit,file,addedLines,deletedLines