import java
import pruning

from Interface interfaceid
where isAnnotType(interfaceid)
  and isSourceLocation(interfaceid.getLocation())
select interfaceid