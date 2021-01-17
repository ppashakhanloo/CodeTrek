import java
import pruning

from Class classid, LocalClassDeclStmt parent
where isLocalClass(classid, parent)
  and isSourceLocation(classid.getLocation())
select classid, parent