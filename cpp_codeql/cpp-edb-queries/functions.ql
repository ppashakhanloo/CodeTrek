import cpp

from Function id,string name,int kind
where functions(id,name,kind)
select id, name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),kind