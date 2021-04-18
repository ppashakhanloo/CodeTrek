import python

from AliasList id,Import parent
where py_alias_lists(id,parent)
and parent.getScope().inSource()
select id,parent
