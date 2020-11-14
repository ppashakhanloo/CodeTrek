//// from Types.qll

predicate referenceType(Type t) {
  derivedtypes(t,_,2,_) or 
  derivedtypes(t,_,8,_)
}

predicate simpleType(Type t) {
  not exists(int i | derivedtypes(t,_,i,_) and (i in [1..5] or i = 8)) and
  not (usertypes(t,_,5))
}

class Type extends @type {
  string toString() { none() }
}

int getArraySize(Type array) {
  arraysizes(array, result, _, _) and
  derivedtypes(array, _, 4, _)
}

predicate hasArraySize(Type array) {
  exists(getArraySize(array))
}

Type getUnspecifiedType(Type t) {
  simpleType(t) and result = t or
  result = getArrayUnspecifiedType(t) or
  result = getGNUVectorUnspecifiedType(t) or
  result = getPointerUnspecifiedType(t) or
  result = getReferenceUnspecifiedType(t) or
  result = getSpecifiedUnspecifiedType(t) or
  result = getTypedefUnspecifiedType(t)
}

Type getPointerUnspecifiedType(Type t) {
  exists(Type tBase, Type resultBase |
    derivedtypes(t,_,1, tBase) and
    derivedtypes(result,_,1, resultBase) and
    resultBase = getUnspecifiedType(tBase) 
  )
}


Type getReferenceUnspecifiedType(Type t) {
  exists(Type tBase, Type resultBase |
    referenceType(t) and referenceType(result) and
    derivedtypes(t,_,_, tBase) and
    derivedtypes(result,_,_, resultBase) and
    resultBase = getUnspecifiedType(tBase) 
  )
}

Type getSpecifiedUnspecifiedType(Type t) {
  exists(Type base |
    derivedtypes(t,_,3, base) and
    result = getUnspecifiedType(base) 
  )
}

Type getArrayUnspecifiedType(Type t) {
  exists(Type tBase, Type resultBase |
    derivedtypes(t,_,4, tBase) and
    derivedtypes(result,_,4, resultBase) and
    resultBase = getUnspecifiedType(tBase) 
  ) and (
    (hasArraySize(t) or hasArraySize(result)) implies getArraySize(t) = getArraySize(result)
  )
}

Type getGNUVectorUnspecifiedType(Type t) {
  exists(Type tBase, Type resultBase |
    derivedtypes(t,_,5, tBase) and
    derivedtypes(result,_,5, resultBase) and
    resultBase = getUnspecifiedType(tBase) 
  )
}

Type getTypedefUnspecifiedType(Type t) {
  exists(Type base |
    usertypes(t,_,5) and
    typedefbase(t,base) and
    result = getUnspecifiedType(base) 
  )
}

from Type t, Type u
where if exists(getUnspecifiedType(t))
      then u = getUnspecifiedType(t)
      else // We don't have a real unspecified type, but some type
           // is probably better than no type
           u = t
select t, u
