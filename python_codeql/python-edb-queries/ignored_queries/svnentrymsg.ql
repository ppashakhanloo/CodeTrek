import python
import external.VCS

from Commit id,string message
where svnentrymsg(id,message)
select id,
       message.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")