import python
import pruning

from ParameterList id,Function parent
where py_parameter_lists(id,parent)
  and isSourceLocation(id.getAnItem().getLocation())
select id,parent
