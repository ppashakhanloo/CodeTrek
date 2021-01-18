import java
import pruning

from Member memberid
where isParameterized(memberid)
  and isSourceLocation(memberid.getLocation())
select memberid