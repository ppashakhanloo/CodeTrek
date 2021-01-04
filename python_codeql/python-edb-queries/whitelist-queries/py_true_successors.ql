import python
import pruning

from Object predecessor,Object successor
where py_true_successors(predecessor,successor)
  and forall (AstNode origin | origin = predecessor.getOrigin() |
        isSourceLocation(origin.getLocation()))
select predecessor,successor