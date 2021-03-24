import python

from Object predecessor,Object successor
where py_true_successors(predecessor,successor)
  and predecessor.getOrigin().getScope().inSource()
  and successor.getOrigin().getScope().inSource()
select predecessor,successor