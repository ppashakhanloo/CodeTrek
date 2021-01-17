import java
import extra_classes
import pruning

from MyLocatable locatableid, Location id
where hasLocation(locatableid, id)
  and isSourceLocation(id)
select locatableid, id