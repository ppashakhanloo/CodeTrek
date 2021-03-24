import python
import restrict_boundaries

from SsaVariable id,Variable var
where py_ssa_var(id,var)
  and var.getScope().inSource()
  and isInBounds(var.getScope())
select id,var