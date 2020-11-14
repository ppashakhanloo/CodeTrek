class TopLevel extends @toplevel {
  string toString() { none() }
}

from TopLevel tl
where toplevels(tl, 4)
select tl
