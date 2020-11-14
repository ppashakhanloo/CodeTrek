
class StmtBlock      extends @stmt_block       { string toString() { none() } }
class StmtSwitch     extends @stmt_switch      { string toString() { none() } }
class StmtSwitchCase extends @stmt_switch_case { string toString() { none() } }

from StmtSwitch s, StmtBlock b, StmtSwitchCase c,
     int block_index, int switch_index
where stmtparents(c, block_index, b)
  and stmtparents(b, 1, s)
  and switch_index = count(StmtSwitchCase c2 |
                           exists(int bi2 | stmtparents(c2, bi2, b) and
                                            bi2 < block_index))
select s, switch_index, c

