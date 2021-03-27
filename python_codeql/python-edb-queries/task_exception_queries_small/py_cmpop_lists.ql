import python

from CmpopList id,Compare parent
where py_cmpop_lists(id,parent)
  and parent.getScope().inSource()
select id,parent