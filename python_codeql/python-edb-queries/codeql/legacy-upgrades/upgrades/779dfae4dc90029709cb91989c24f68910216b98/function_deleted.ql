
class Location extends @location_default { string toString() { none() } }
class File extends @file { string toString() { none() } }
class Block extends @stmt_block { string toString() { none() } }
class StmtLocation extends @location_stmt { string toString() { none() } }
class Function extends @function { string toString() { none() } }
class FunctionDeclarationEntry extends @fun_decl {
  string toString() { none() }
}

predicate isSpecialMember(Function f) {
  functions(f,_,2) or functions(f,_,3) or functions(f,"operator=",_)
}

from FunctionDeclarationEntry d, Function f, Location l
where fun_decls(d,f,_,_,l)
  and fun_def(d)
  and member(_,_,f)
  and not exists(Block b, StmtLocation bl, File file |
                 blockscope(b,f) and
                 stmts(b,_,bl) and
                 locations_stmt(bl,file,_,_,_,_) and
                 locations_default(l,file,_,_,_,_))
  and not function_template_argument(f,_,_)
  and not exists(FunctionDeclarationEntry d2 |
                 fun_decls(d2,f,_,_,_) and d != d2)
  and not exists(int sl, int sc, int el, int ec |
        locations_default(l,_,sl,sc,el,ec) and sl = el and ec <= sc + 1)
  and isSpecialMember(f)
select f

