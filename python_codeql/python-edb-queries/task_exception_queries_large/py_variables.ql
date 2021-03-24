import python
import extra_classes
import restrict_boundaries

from Variable id,VariableParent parent
where py_variables(id,parent)
  and id.getScope().inSource()
  and isInBounds(id.getScope())
select id,parent