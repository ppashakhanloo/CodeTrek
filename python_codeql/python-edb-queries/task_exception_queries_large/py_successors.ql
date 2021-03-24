import python
import restrict_boundaries

from Object predecessor,Object successor
where py_successors(predecessor,successor)
  and predecessor.getOrigin().getScope().inSource()
  and successor.getOrigin().getScope().inSource()
  and isInBounds(predecessor.getOrigin().getScope())
  and isInBounds(successor.getOrigin().getScope())
select predecessor,successor