import python
import external.VCS

from Commit id,Container file,string action
where svnaffectedfiles(id,file,action)
select id,file,action