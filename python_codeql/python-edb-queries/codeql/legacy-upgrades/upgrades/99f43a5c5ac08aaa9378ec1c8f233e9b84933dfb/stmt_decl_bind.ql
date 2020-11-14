
class StmtDecl extends @stmt_decl { string toString() { none() } }
class Decl extends @declaration { string toString() { none() } }

from StmtDecl sd, Decl d
where stmt_decl_bind(sd, d)
select sd, 0, d

