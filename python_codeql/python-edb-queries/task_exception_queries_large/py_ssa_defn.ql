import python
import restrict_boundaries

from SsaVariable id,Object node
where py_ssa_defn(id,node)
  and id.getVariable().getScope().inSource()
  and node.getOrigin().getScope().inSource()
  and isInBounds(id.getVariable().getScope())
  and isInBounds(node.getOrigin().getScope())
select id,node