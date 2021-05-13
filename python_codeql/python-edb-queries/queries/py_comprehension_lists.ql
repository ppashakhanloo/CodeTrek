import python
import pruning

from ComprehensionList id,ListComp parent
where py_comprehension_lists(id,parent)
  and isSourceLocation(parent.getLocation())
select id,parent
