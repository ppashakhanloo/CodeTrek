import python

from Object predecessor,Object successor
where py_exception_successors(predecessor,successor)
  and forall (AstNode origin | origin = predecessor.getOrigin() |
        origin.getScope().inSource())
select predecessor,successor