import python
import external.ExternalArtifact

from ExternalMetric id,string queryPath,Location location,float value
where externalMetrics(id,queryPath,location,value)
select id,queryPath,location,value