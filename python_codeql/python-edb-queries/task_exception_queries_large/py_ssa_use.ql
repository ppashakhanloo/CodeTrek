import python
import restrict_boundaries

from Object node,SsaVariable var
where py_ssa_use(node,var)
  and node.getOrigin().getScope().inSource()
  and isInBounds(node.getOrigin().getScope())
select node,var