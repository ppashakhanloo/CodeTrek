// assume all user types in an existing db are complete

class MyType extends @usertype {
  string toString() { usertypes(this, result, _) }
}

from MyType t
select t
