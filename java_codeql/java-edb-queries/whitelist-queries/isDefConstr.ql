import java
import pruning

from Constructor constructorid
where isDefConstr(constructorid)
  and isSourceLocation(constructorid.getLocation())
select constructorid