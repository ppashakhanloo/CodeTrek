import python
import extra_classes
import pruning

from MyDictItem id,int kind, MyDictItemList parent, int idx
where py_dict_items(id, kind, parent,idx)
  and isSourceLocation(id.getLocation())
select id,kind,parent,idx