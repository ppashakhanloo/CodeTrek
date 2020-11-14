class MyExpr extends @expr { string toString() { result="expr" } }
class MyType extends @type { string toString() { result="type" } }
class MyElement extends @element { string toString() { result="element" } }

from MyExpr expr, int kind, int newkind, MyType t, int index, MyElement parent
where expressions(expr, kind, t, index, parent)
and if kind=-1 then newkind=106 else newkind=kind
select expr, kind, t, index, parent