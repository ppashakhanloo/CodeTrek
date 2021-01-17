import java
import pruning

from Member memberid
where isRaw(memberid)
  and isSourceLocation(memberid.getLocation())
select memberid