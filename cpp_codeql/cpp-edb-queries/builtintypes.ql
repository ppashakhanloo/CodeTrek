import cpp

from BuiltInType id,string name,int kind,int size,int sign,int alignment
where builtintypes(id,name,kind,size,sign,alignment)
select id,name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),kind,size,sign,alignment