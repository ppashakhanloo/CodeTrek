class Expr extends @expr
{
  string toString() {
    result = ""
  }
}

from Expr e
where 
  e instanceof @c_style_cast or
  e instanceof @reinterpret_cast or
  e instanceof @static_cast or
  e instanceof @const_cast
select e, 0  // Just pretend every cast is a simple cast
