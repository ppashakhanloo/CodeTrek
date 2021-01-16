import java

from Array id, string nodeName, Type elementtypeid, int dimension, Type componenttypeid
where arrays(id, nodeName, elementtypeid, dimension, componenttypeid)
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"), elementtypeid, dimension, componenttypeid