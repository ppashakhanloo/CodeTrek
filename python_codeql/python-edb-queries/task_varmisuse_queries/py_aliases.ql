import python

from Alias id,AliasList parent,int idx
where py_aliases(id,parent,idx)
and parent.getParent().getScope().inSource()
select id,parent,idx
