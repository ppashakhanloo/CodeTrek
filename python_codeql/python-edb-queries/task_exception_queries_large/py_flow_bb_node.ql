import python
import restrict_boundaries

from Object flownode,AstNode realnode,Object basicblock,int index
where py_flow_bb_node(flownode,realnode,basicblock,index)
  and flownode.getOrigin().getScope().inSource()
  and realnode.getScope().inSource()
  and isInBounds(flownode.getOrigin().getScope())
  and isInBounds(realnode.getScope())
select flownode,realnode,basicblock,index