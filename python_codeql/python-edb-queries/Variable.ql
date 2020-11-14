/**
 * @kind path-problem
 * @id your-query-id
 */

import python

from Variable id
where py_variables(id, _)
select id, id.getScope()
/* id.getId() */