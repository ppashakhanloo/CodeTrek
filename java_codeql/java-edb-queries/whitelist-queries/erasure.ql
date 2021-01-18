import java
import pruning

from Member memberid, Member erasureid
where erasure(memberid, erasureid)
  and isSourceLocation(memberid.getLocation())
select memberid, erasureid