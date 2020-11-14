
class MyExpr extends @expr {

  /** a printable representation of this expression */
  string toString() { result = "Expression" }
}

class MyCall extends MyExpr, @call {
  
  /** The `i`th argument of this call. */
  MyExpr getArgument(int i) { expressions(result,_,_,i,this) and i >= 0 }
  
   /** the constructor referenced by this initializer */
  MyCallable getTarget() { expr_call(this,result) }
  
  /** a printable representation of this call */
  override string toString() { result = "call" }
}

class MyParameter extends @parameter 
{
  string toString() { result = "Param" }
  
  /** whether this is a reference parameter */
  predicate isRef() { params(this,_,_,_,1,_,_) }
  
  /** whether this is an output parameter */
  predicate isOut() { params(this,_,_,_,2,_,_) }
  
}

class MyCallable extends @callable {
  
  /** The nth parameter of this artifact. */
  MyParameter getParameter(int n) { params(result,_,_,n,_,this,_) }
 
  string toString() { result = "Callable" }
}

pragma[noinline]
predicate findArgument(int argument, MyCallable target, MyExpr arg) {
  exists(MyCall call |
    target = call.getTarget() and 
    arg = call.getArgument(argument) )
}

int argumentMode(MyExpr arg)
{
  exists(int argument, MyParameter param, MyCallable target |
    findArgument(argument, target, arg) and
    param = target.getParameter(argument)
    |
    if param.isOut() then result=2
    else if param.isRef() then result=1
    else result=0
    )
}

from MyExpr arg
select arg, argumentMode(arg)
