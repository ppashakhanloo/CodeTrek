import python
import extra_classes
import pruning

from Variable id,VariableParent parent
where py_variables(id,parent)
  and isSourceLocation(id.getScope().getLocation())
select id,parent