import python

from SsaVariable id,Object node
where py_ssa_defn(id,node)
  and id.getVariable().getScope().inSource()
select id,node