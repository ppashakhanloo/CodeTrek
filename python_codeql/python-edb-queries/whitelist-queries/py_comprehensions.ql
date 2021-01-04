import python
import pruning

from Comprehension id,ComprehensionList parent,int idx
where py_comprehensions(id,parent,idx)
  and isSourceLocation(id.getLocation())
select id,parent,idx