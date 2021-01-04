import python
import pruning

from Object node,SsaVariable var
where py_ssa_use(node,var)
  and isSourceLocation(var.getLocation())
select node,var