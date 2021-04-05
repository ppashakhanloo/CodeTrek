import python

from ComprehensionList id,ListComp parent
where py_comprehension_lists(id,parent)
  and parent.getScope().inSource()
select id,parent