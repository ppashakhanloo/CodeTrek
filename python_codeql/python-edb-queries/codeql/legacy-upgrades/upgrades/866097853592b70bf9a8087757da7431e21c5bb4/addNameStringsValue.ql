
class NamedExprOrStmt extends @namedexprorstmt {
  string toString() { namestrings(result, this) }
}

class Literal extends NamedExprOrStmt, @literal {}

class BooleanLiteral extends Literal,@booleanliteral {}
class IntegerLiteral extends Literal,@integerliteral {}
class LongLiteral extends Literal,@longliteral {}
class FloatingPointLiteral extends Literal,@floatingpointliteral {}
class DoubleLiteral extends Literal,@doubleliteral {}
class CharacterLiteral extends Literal,@characterliteral {}
class StringLiteral extends Literal,@stringliteral {}
class NullLiteral extends Literal,@nullliteral {}

class CharOrStringLit extends Literal {
  CharOrStringLit() {
    this instanceof CharacterLiteral or this instanceof StringLiteral
  }
}

class FloatOrDoubleLit extends Literal {
  FloatOrDoubleLit() {
    this instanceof FloatingPointLiteral or this instanceof DoubleLiteral
  }
}

class IntOrLongLit extends Literal {
  IntOrLongLit() {
    this instanceof IntegerLiteral or this instanceof LongLiteral
  }
}

string convertString(string name, CharOrStringLit parent) {
  namestrings(name, parent) and
  exists(string trimmed, string joined |
    trimmed = name.substring(1, name.length() - 1) and
    joined = trimmed.regexpReplaceAll("([^\\\\])\"\\s*\\+\\s*\"", "$1") and
    // this conversion doesn't handle '\b', '\f', or '\uXXXX'
    result =
      joined
        .regexpReplaceAll("\\\\t", "\t")
        .regexpReplaceAll("\\\\n", "\n")
        .regexpReplaceAll("\\\\r", "\r")
        .regexpReplaceAll("\\\\'", "'")
        .regexpReplaceAll("\\\\\"", "\"")
        .regexpReplaceAll("\\\\\\\\", "\\")
  )
}

string convertPartialFloatDouble(string name, FloatOrDoubleLit parent) {
  namestrings(name, parent) and
  (
  	result = name.regexpReplaceAll("^([0-9]+)(\\.)?[fdFD]$", "$1.0") or
    result = name.regexpReplaceAll("^\\.([0-9]+)[fdFD]$", "0.$1")
  ) and
  result != name
}

string convertFloatAndDouble(string name, FloatOrDoubleLit parent) {
  namestrings(name, parent) and
  if exists(convertPartialFloatDouble(name, parent)) then
  	result = convertPartialFloatDouble(name, parent)
  else
  	result = name.regexpReplaceAll("[fdFD]", "")
}

string convertCommonHexAndOctal(string name, IntOrLongLit parent) {
  namestrings(name, parent) and
  exists(string l | l = name.toLowerCase().replaceAll("_", "") |
    l.regexpMatch("(0x0|0x00|0x0000|0x00000000|00)l?")  and result = "0" or
    l.regexpMatch("(0x1|0x01|0x0001|0x00000001|01)l?")  and result = "1" or
    l.regexpMatch("(0x2|0x02|0x0002|0x00000002|02)l?")  and result = "2" or
    l.regexpMatch("(0x4|0x04|0x0004|0x00000004|04)l?")  and result = "4" or
    l.regexpMatch("(0x8|0x08|0x0008|0x00000008|010)l?") and result = "8" or
    l.regexpMatch("(0x10|0x0010|0x00000010|020)l?")     and result = "16" or
    l.regexpMatch("(0x20|0x0020|0x00000020|040)l?")     and result = "32" or
    l.regexpMatch("(0x40|0x0040|0x00000040|0100)l?")    and result = "64" or
    l.regexpMatch("(0x80|0x0080|0x00000080|0200)l?")    and result = "128" or
    l.regexpMatch("(0x0100|0x00000100|0400)l?")         and result = "256" or
    l.regexpMatch("(0x0200|0x00000200|01000)l?")        and result = "512" or
    l.regexpMatch("(0x0400|0x00000400|02000)l?")        and result = "1024" or
    l.regexpMatch("(0x0800|0x00000800|04000)l?")        and result = "2048" or
    l.regexpMatch("(0x1000|0x00001000)l?")              and result = "4096" or
    l.regexpMatch("(0x2000|0x00002000)l?")              and result = "8192" or
    l.regexpMatch("(0x4000|0x00004000)l?")              and result = "16384" or
    l.regexpMatch("(0x8000|0x00008000)l?")              and result = "32768" or
    l.regexpMatch("0x00010000l?")                       and result = "65536" or
    l.regexpMatch("0x00020000l?")                       and result = "131072" or
    l.regexpMatch("0x00040000l?")                       and result = "262144" or
    l.regexpMatch("0x00080000l?")                       and result = "524288" or
    l.regexpMatch("0x00100000l?")                       and result = "1048576" or
    l.regexpMatch("0x00200000l?")                       and result = "2097152" or
    l.regexpMatch("0x00400000l?")                       and result = "4194304" or
    l.regexpMatch("0x00800000l?")                       and result = "8388608" or
    l.regexpMatch("0x01000000l?")                       and result = "16777216" or
    l.regexpMatch("0x02000000l?")                       and result = "33554432" or
    l.regexpMatch("0x04000000l?")                       and result = "67108864" or
    l.regexpMatch("0x08000000l?")                       and result = "134217728" or
    l.regexpMatch("0x10000000l?")                       and result = "268435456" or
    l.regexpMatch("0x20000000l?")                       and result = "536870912" or
    l.regexpMatch("0x40000000l?")                       and result = "1073741824" or
    l.regexpMatch("0x80000000l?") and
      (if parent instanceof LongLiteral then result = "2147483648" else result = "-2147483648") or

    l.regexpMatch("(0x3|0x03|0x0003|0x00000003|03)l?")  and result = "3" or
    l.regexpMatch("(0x7|0x07|0x0007|0x00000007|07)l?")  and result = "7" or
    l.regexpMatch("(0xf|0x0f|0x000f|0x0000000f|017)l?") and result = "15" or
    l.regexpMatch("(0x1f|0x001f|0x0000001f|037)l?")     and result = "31" or
    l.regexpMatch("(0x3f|0x003f|0x0000003f|077)l?")     and result = "63" or
    l.regexpMatch("(0x7f|0x007f|0x0000007f|0177)l?")    and result = "127" or
    l.regexpMatch("(0xff|0x00ff|0x000000ff|0377)l?")    and result = "255" or
    l.regexpMatch("(0x01ff|0x000001ff|0777)l?")         and result = "511" or
    l.regexpMatch("(0x03ff|0x000003ff|01777)l?")        and result = "1023" or
    l.regexpMatch("(0x07ff|0x000007ff|03777)l?")        and result = "2047" or
    l.regexpMatch("(0x0fff|0x00000fff|07777)l?")        and result = "4095" or
    l.regexpMatch("(0x1fff|0x00001fff)l?")              and result = "8191" or
    l.regexpMatch("(0x3fff|0x00003fff)l?")              and result = "16383" or
    l.regexpMatch("(0x7fff|0x00007fff)l?")              and result = "32767" or
    l.regexpMatch("(0xffff|0x0000ffff)l?")              and result = "65535" or
    l.regexpMatch("0x0001ffffl?")                       and result = "131071" or
    l.regexpMatch("0x0003ffffl?")                       and result = "262143" or
    l.regexpMatch("0x0007ffffl?")                       and result = "524287" or
    l.regexpMatch("0x000fffffl?")                       and result = "1048575" or
    l.regexpMatch("0x001fffffl?")                       and result = "2097151" or
    l.regexpMatch("0x003fffffl?")                       and result = "4194303" or
    l.regexpMatch("0x007fffffl?")                       and result = "8388607" or
    l.regexpMatch("0x00ffffffl?")                       and result = "16777215" or
    l.regexpMatch("0x01ffffffl?")                       and result = "33554431" or
    l.regexpMatch("0x03ffffffl?")                       and result = "67108863" or
    l.regexpMatch("0x07ffffffl?")                       and result = "134217727" or
    l.regexpMatch("0x0fffffffl?")                       and result = "268435455" or
    l.regexpMatch("0x1fffffffl?")                       and result = "536870911" or
    l.regexpMatch("0x3fffffffl?")                       and result = "1073741823" or
    l.regexpMatch("0x7fffffffl?")                       and result = "2147483647" or
    l.regexpMatch("0xffffffffl?") and
      (if parent instanceof LongLiteral then result = "4294967295" else result = "-1")
  )
}

string convertUnderscores(string name, IntOrLongLit parent) {
  namestrings(name, parent) and
  name.regexpMatch("[0-9]+_[0-9][0-9_]*[lL]?") and
  result = name.regexpReplaceAll("[_lL]", "")
}

string convertIntAndLong(string name, IntOrLongLit parent) {
  namestrings(name, parent) and
  if exists(convertCommonHexAndOctal(name, parent)) then
    result = convertCommonHexAndOctal(name, parent)
  else if exists(convertUnderscores(name, parent)) then
    result = convertUnderscores(name, parent)
  else
    result = name.regexpReplaceAll("[lL]$", "")
}

string convertName(string name, NamedExprOrStmt parent) {
  namestrings(name, parent) and
  (
    not parent instanceof Literal and result = "" or
    result = convertString(name, parent) or
    result = convertFloatAndDouble(name, parent) or
    result = convertIntAndLong(name, parent)
  )
}

from string name, string value, NamedExprOrStmt parent
where
  namestrings(name, parent) and
  (
    if exists(convertName(name, parent)) then
      value = convertName(name, parent)
    else
      value = name
  )
select name, value, parent
