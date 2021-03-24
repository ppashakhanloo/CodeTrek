import python
import extra_classes
import restrict_boundaries

from MyDictItemList dil, MyDictItemListParent dilp
where py_dict_item_lists(dil, dilp)
  and dil.getAnItem().getScope().inSource()
  and isInBounds(dil.getAnItem().getScope())
select dil, dilp