import java
import pruning

from Parameter param
where isVarargsParam(param)
  and isSourceLocation(param.getLocation())
select param