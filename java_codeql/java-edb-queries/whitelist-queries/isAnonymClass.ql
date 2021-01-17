import java
import pruning

from Class classid, ClassInstanceExpr parent
where isAnonymClass(classid, parent)
  and isSourceLocation(classid.getLocation())
select classid, parent