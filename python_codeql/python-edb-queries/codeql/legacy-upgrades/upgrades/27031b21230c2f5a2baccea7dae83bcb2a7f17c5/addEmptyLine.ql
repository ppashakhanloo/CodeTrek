class TopLevel extends @toplevel {
  string toString() { result = "toplevel" }
}

class JSParseError extends @js_parse_error {
  string toString() {
    jsParseErrors(this, _, result)
  }
}

from JSParseError err, TopLevel tl, string msg
where jsParseErrors(err, tl, msg)
select err, tl, msg, ""
