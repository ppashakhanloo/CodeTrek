import python
import extra_classes

from Location id, LocationParent parent
where py_locations(id,parent)
  and not id.getFile().inStdlib()
  and not id.getFile().isImportRoot()
select id,parent