import java
import extra_classes
import pruning

from Type argumentid, int pos, MyTypeOrCallable parentid
where typeArgs(argumentid, pos, parentid)
  and isSourceLocation(argumentid.getLocation())
select argumentid, pos, parentid