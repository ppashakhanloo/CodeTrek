import java
import pruning

from Field fieldid
where isEnumConst(fieldid)
  and isSourceLocation(fieldid.getLocation())
select fieldid