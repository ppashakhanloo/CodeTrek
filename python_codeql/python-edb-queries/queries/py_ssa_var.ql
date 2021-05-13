import python
import pruning

from SsaVariable id,Variable var
where py_ssa_var(id,var)
  and isSourceLocation(id.getLocation())
select id,var
