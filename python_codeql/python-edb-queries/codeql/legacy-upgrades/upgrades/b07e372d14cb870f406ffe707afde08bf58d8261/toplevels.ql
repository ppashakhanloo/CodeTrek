class TopLevel extends @toplevel {
  string toString() { none() }
}

from TopLevel tl, int oldKind, int newKind
where toplevels(tl, oldKind) and
      // map toplevel kind 4 to 0, preserve all others
      (if oldKind = 4 then
         newKind = 0
       else
         newKind = oldKind)
select tl, newKind
