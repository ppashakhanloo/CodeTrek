import python

from Object flownode,AstNode realnode,Object basicblock,int index
where py_flow_bb_node(flownode,realnode,basicblock,index)
  and realnode.getScope().inSource()
select flownode,realnode,basicblock,index