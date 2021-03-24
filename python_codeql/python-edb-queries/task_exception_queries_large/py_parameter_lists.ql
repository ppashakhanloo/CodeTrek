import python
import restrict_boundaries

from ParameterList id,Function parent
where py_parameter_lists(id,parent)
  and parent.inSource()
  and isInBounds(parent)
select id,parent