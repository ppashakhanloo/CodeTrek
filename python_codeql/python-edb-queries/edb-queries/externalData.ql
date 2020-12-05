import python
import external.ExternalArtifact

from ExternalData id,string queryPath,int column,string data
where externalData(id,queryPath,column,data)
select id,queryPath,column,data