import python

from Object node,SsaVariable var
where py_ssa_use(node,var)
  and node.getOrigin().getScope().inSource()
  and var.getVariable().getScope().inSource()
select node,var