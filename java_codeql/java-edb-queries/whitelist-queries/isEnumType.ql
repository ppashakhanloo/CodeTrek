import java
import pruning

from Class classid
where isEnumType(classid)
  and isSourceLocation(classid.getLocation())
select classid