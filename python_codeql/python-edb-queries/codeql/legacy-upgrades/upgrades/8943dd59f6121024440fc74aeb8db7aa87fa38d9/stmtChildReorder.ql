import Lib

from Stmt s, int kind, Element parent, int old_idx, Element callable, int new_idx
where stmts(s, kind, parent, old_idx, callable) and
      // condense statement indices inside block statements
      if parent instanceof BlockStmt then
        exists (int rnk |
          // find the (one-based) rank of this statement within its block
          old_idx = rank[rnk](Stmt ss, int kkind, int i | stmts(ss, kkind, parent, i, callable) | i) and
          // its statement index should be one less
          new_idx = rnk-1
        )
      else
        // move for loop bodies to position 2 (see Lib.qll)
        parent.move(old_idx, parent, new_idx)
select s, kind, parent, new_idx, callable