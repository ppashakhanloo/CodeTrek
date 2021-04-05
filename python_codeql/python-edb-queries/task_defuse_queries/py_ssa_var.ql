import python

from SsaVariable id,Variable var
where py_ssa_var(id,var)
  and var.getScope().inSource()
  and id.getVariable().getScope().inSource()
select id,var