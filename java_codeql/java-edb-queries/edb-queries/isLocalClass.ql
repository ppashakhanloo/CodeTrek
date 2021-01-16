import java

from Class classid, LocalClassDeclStmt parent
where isLocalClass(classid, parent)
select classid, parent