import python
import restrict_boundaries

from ComprehensionList id,ListComp parent
where py_comprehension_lists(id,parent)
  and parent.getScope().inSource()
  and isInBounds(parent.getScope())
select id,parent