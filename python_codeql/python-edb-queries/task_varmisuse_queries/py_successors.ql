import python

from Object predecessor,Object successor
where py_successors(predecessor,successor)
  and (predecessor.getOrigin().getScope().inSource()
    or successor.getOrigin().getScope().inSource())
select predecessor,successor