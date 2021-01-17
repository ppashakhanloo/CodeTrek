import java
import pruning

from Type id1, ClassOrInterface id2
where extendsReftype(id1, id2)
  and isSourceLocation(id1.getLocation())
select id1, id2