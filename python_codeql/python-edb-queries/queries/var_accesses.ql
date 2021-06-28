import python
import pruning

from Name n, LocalVariable lv
where isSourceLocation(n.getLocation())
  and not n.isConstant()
  and n.getVariable() = lv
select n, n.getCtx(), n.getVariable()
