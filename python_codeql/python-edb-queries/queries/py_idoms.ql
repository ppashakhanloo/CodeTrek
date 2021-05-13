import python
import pruning

from Object node,Object immediate_dominator
where py_idoms(node,immediate_dominator)
  and forall (AstNode origin | origin = node.getOrigin() |
        isSourceLocation(origin.getLocation()))
select node,immediate_dominator
