/**
 * @kind path-problem
 * @id your-query-id
 */

import python

from Stmt id, int kind, StmtList parent
where py_stmts(id, kind, parent, _)
select id, id.getLocation()
