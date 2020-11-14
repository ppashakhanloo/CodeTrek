/**
 * @kind path-problem
 * @id your-query-id
 */

import python

from Location loc, int beginLine, int beginColumn, int endLine, int endColumn
where locations_default(loc, _, beginLine, beginColumn, endLine, endColumn) or
      locations_ast(loc, _, beginLine, beginColumn, endLine, endColumn)
select loc, beginLine, beginColumn, endLine, endColumn