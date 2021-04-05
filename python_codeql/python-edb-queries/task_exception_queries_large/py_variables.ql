import python
import extra_classes

from Variable id,VariableParent parent
where py_variables(id,parent)
  and id.getScope().inSource()
select id,parent