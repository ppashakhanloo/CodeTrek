
class PointerIshDerivedType extends @derivedtype {
    PointerIshDerivedType() {
        derivedtypes(this,_,1,_) or
        derivedtypes(this,_,2,_) or
        derivedtypes(this,_,6,_) or
        derivedtypes(this,_,7,_) or
        derivedtypes(this,_,8,_) or
        derivedtypes(this,_,10,_)
    }
    int getSize() {
        if derivedtypes(this,_,7,_) then result = 1 else pointersize(result)
    }
    string toString() { none() }
}

from PointerIshDerivedType t
select t, t.getSize()

