import python

from Comprehension id,ComprehensionList parent,int idx
where py_comprehensions(id,parent,idx)
  and id.getScope().inSource()
select id,parent,idx