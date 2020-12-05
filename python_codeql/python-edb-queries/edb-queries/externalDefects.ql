import python
import external.ExternalArtifact

from ExternalDefect id,string queryPath,Location location,string message,float severity
where externalDefects(id,queryPath,location,message,severity)
select id,queryPath,location,message,severity