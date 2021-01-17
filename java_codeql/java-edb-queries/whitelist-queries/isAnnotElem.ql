import java
import pruning

from Method methodid
where isAnnotElem(methodid)
  and isSourceLocation(methodid.getLocation())
select methodid