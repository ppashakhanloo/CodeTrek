import python

from Object node,Object immediate_dominator
where py_idoms(node,immediate_dominator)
  and node.getOrigin().getScope().inSource()
  and immediate_dominator.getOrigin().getScope().inSource()
select node,immediate_dominator