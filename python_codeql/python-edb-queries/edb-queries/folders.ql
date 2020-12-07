import python

from Container id,string name,string simple
where folders(id,name,simple)
select id,
       name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       simple.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")