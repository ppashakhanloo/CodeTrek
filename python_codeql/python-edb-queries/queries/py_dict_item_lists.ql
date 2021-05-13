import python
import extra_classes
import pruning

from MyDictItemList dil, MyDictItemListParent dilp
where py_dict_item_lists(dil, dilp)
  and isSourceLocation(dil.getAnItem().getLocation())
select dil, dilp
