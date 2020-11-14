class CfgNode extends @cfg_node {
  CfgNode() { not this instanceof @class }
  string toString() { none() }
}

from CfgNode pred, CfgNode succ
where successor(pred, succ)
  or exists (@class cl | successor(pred, cl) and successor(cl, succ))
select pred, succ
