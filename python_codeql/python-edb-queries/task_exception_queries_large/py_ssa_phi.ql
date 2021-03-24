import python
import restrict_boundaries

from SsaVariable phi,SsaVariable arg
where py_ssa_phi(phi,arg)
  and phi.getVariable().getScope().inSource()
  and arg.getVariable().getScope().inSource()
  and isInBounds(phi.getVariable().getScope())
  and isInBounds(arg.getVariable().getScope())
select phi,arg