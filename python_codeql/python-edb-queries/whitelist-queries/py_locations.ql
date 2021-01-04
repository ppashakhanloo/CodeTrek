import python
import extra_classes
import pruning

from Location id, LocationParent parent
where py_locations(id,parent)
  and isSourceLocation(id)
select id,parent