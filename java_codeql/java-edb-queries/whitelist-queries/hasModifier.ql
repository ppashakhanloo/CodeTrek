import java
import pruning

from Modifiable id1, Modifier id2
where hasModifier(id1, id2)
  and isSourceLocation(id1.getLocation())
select id1, id2