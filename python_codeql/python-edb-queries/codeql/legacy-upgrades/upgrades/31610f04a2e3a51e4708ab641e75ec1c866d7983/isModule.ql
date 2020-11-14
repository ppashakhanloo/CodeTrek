class TopLevel extends @toplevel {
  string toString() { result = "toplevel" }
}

from TopLevel tl
where scopenodes(tl, _)
select tl
