import java

from Stmt id, int kind, StmtParent parent, int idx, Callable bodydecl
where stmts(id, kind, parent, idx, bodydecl)
select id, kind, parent, idx, bodydecl