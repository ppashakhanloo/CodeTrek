import python
import restrict_boundaries

from Object node,Object immediate_dominator
where py_idoms(node,immediate_dominator)
  and node.getOrigin().getScope().inSource()
  and immediate_dominator.getOrigin().getScope().inSource()
  and isInBounds(node.getOrigin().getScope())
  and isInBounds(immediate_dominator.getOrigin().getScope())
select node,immediate_dominator