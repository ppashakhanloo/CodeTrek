import python
import pruning

from Object flownode,AstNode realnode,Object basicblock,int index
where py_flow_bb_node(flownode,realnode,basicblock,index)
  and isSourceLocation(realnode.getLocation())
select flownode,realnode,basicblock,index