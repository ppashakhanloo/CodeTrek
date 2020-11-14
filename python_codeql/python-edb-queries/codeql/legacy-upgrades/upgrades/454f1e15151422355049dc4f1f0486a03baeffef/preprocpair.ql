abstract class Directive extends @preprocdirect {
  string toString() { result = "pp-directive" }
  predicate hasLocationInfo(string path, int sl, int sc, int el, int ec) {
    exists(@location_default loc, @file f |
      preprocdirects(this, _, loc) and
      locations_default(loc, f, sl, sc, el, ec) and
      files(f, path, _, _, _)
    )
  }
}

class IfLike extends Directive {
  IfLike() { this instanceof @ppd_if or
             this instanceof @ppd_ifdef or
             this instanceof @ppd_ifndef }
}

class ElseLike extends Directive {
  ElseLike() { this instanceof @ppd_else or
               this instanceof @ppd_elif }
}

class EndLike extends Directive {
  EndLike() { this instanceof @ppd_endif }
}

predicate lineEvent(string path, int line, Directive pd) {
  pd.hasLocationInfo(path, line, _, _, _)
}

predicate indexEvent(string path, int index, Directive pd) {
  lineEvent(path, rank[index](int line | lineEvent(path, line, _)), pd)
}

predicate stackTopAfter(string path, int index, Directive top) {
  (
    index = 0 and indexEvent(path, 1, top) /* The choice of top here is arbitrary, and is just to ensure that the stack is never empty. */
  )
  or exists(int prev, Directive otop, Directive cur |
    stackTopAfter(path, prev, otop) and
    index = prev + 1 and
    indexEvent(path, index, cur)
  |
    (cur instanceof IfLike and top = cur) or
    (cur instanceof ElseLike and top = otop) or
    (cur instanceof EndLike and exists(int i | indexEvent(path, i, otop) | stackTopAfter(path, i-1, top)))
  )
}

predicate pair(IfLike iflike, ElseLike elselike) {
  exists(string path, int index |
    indexEvent(path, index, elselike) and
    stackTopAfter(path, index, iflike)
  )
}

from Directive iflike, Directive elike
where pair(iflike, elike)
   or preprocpair(iflike, elike)
select iflike, elike
