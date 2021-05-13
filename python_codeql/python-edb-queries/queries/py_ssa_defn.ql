import python
import pruning

from SsaVariable id,Object node
where py_ssa_defn(id,node)
  and isSourceLocation(id.getLocation())
select id,node
