import python
import extra_classes
import restrict_boundaries

from MyDictItem id,int kind, MyDictItemList parent, int idx
where py_dict_items(id, kind, parent,idx)
  and id.getScope().inSource()
  and isInBounds(id.getScope())
select id,kind,parent,idx