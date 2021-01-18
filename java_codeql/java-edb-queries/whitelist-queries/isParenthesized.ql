import java
import pruning

from Expr id, int parentheses
where isParenthesized(id, parentheses)
  and isSourceLocation(id.getLocation())
select id, parentheses