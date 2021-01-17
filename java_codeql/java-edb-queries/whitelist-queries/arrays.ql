import java
import pruning

from Array id, string nodeName, Type elementtypeid, int dimension, Type componenttypeid
where arrays(id, nodeName, elementtypeid, dimension, componenttypeid)
  and isSourceLocation(id.getLocation())
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"), elementtypeid, dimension, componenttypeid