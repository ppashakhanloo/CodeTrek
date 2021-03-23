import python

from Object node,Object immediate_dominator
where py_idoms(node,immediate_dominator)
  and forall (AstNode origin | origin = node.getOrigin() |
        origin.getScope().inSource())
select node,immediate_dominator