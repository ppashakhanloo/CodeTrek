import python
import external.ExternalArtifact

from ExternalMetric id,string queryPath,Location location,float value
where externalMetrics(id,queryPath,location,value)
select id,
       queryPath.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       location,
       value