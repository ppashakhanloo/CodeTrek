import python
import pruning

from SsaVariable phi,SsaVariable arg
where py_ssa_phi(phi,arg)
  and isSourceLocation(phi.getLocation())
select phi,arg
