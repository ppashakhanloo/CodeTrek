import python

from SsaVariable phi,SsaVariable arg
where py_ssa_phi(phi,arg)
  and (phi.getVariable().getScope().inSource()
    or arg.getVariable().getScope().inSource())
select phi,arg