import cpp

from DerivedType id,string name,int kind,Type type_id
where derivedtypes(id,name,kind,type_id)
select id,name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),kind,type_id