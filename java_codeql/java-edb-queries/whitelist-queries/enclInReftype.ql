import java
import pruning

from Type child, Type parent
where enclInReftype(child, parent)
  and isSourceLocation(parent.getLocation())
select child, parent