class ArithmeticType2 extends Type2, @builtintype {
  ArithmeticType2() {
    exists(int kind | builtintypes(this,_,kind,_,_) and kind >= 4)
  }
  
  override string toString() { result = "" }
}
class UserType2 extends Type2, @usertype {
}

class Enum2 extends @usertype {
  Enum2() {
    usertypes(this,_,4) or usertypes(this,_,13)
  }
  
  string toString() { result = "" }
}

class FunctionPointerType2 extends @derivedtype {
  FunctionPointerType2() {
    derivedtypes(this,_,6,_)
  }
  
  string toString() { result = "" }
}

class PointerToMemberType2 extends @ptrtomember {
  string toString() { result = "" }
}

class PointerType2 extends @derivedtype {
  PointerType2() { derivedtypes(this,_,1,_) }
  
  string toString() { result = "" }
}

class Type2 extends @type {
  Type2 getUnspecifiedType() { unspecifiedtype(this, result) }
  
  string toString() { result = "" }
}

class ArrayType2 extends @derivedtype {
  ArrayType2() { derivedtypes(this,_,4,_) }
  Type2 getBaseType() { derivedtypes(this,_,_,result) }
  string toString() { result = "" }
}

class ClassDerivation2 extends @derivation {
  Class2 getBaseClass() {
    result = getBaseType().getUnspecifiedType()
  }
  Type2 getBaseType() {
    derivations(this,_,_,result,_)
  }
  Class2 getDerivedClass() {
    derivations(this,result,_,_,_)
  }
  string toString() { result = "" }
}

class Class2 extends UserType2 {
  Class2() {
    usertypes(this,_,1) or usertypes(this,_,2) or usertypes(this,_,3) or usertypes(this,_,6)
    or usertypes(this,_,10) or usertypes(this,_,11) or usertypes(this,_,12)
  }
  Class2 getABaseClass() { this.getADerivation().getBaseClass() = result }
  ClassDerivation2 getADerivation() {
    exists(ClassDerivation2 d | d.getDerivedClass() = this and d = result)
  }
}

class TemplateClass2 extends Class2 {
  TemplateClass2() { usertypes(this,_,6) }
}

class Parameter2 extends Variable2, @parameter {
  override Type2 getType() { params(this,_,_,result) }
}

class Function2 extends @function {
  string getName() { functions(this,result,_) }
  predicate hasName(string name) { name = this.getName() }

  FunctionDeclarationEntry2 getADeclarationEntry() {
    if fun_decls(_,this,_,_,_) then
      declEntry(result)
    else
      exists(Function2 f | function_instantiation(this,f) and fun_decls(result,f,_,_,_))
  }

  private predicate declEntry(FunctionDeclarationEntry2 fde) {
    fun_decls(fde,this,_,_,_) and
    (this.isSpecialization() implies fde.isSpecialization())
  }

  Location2 getADeclarationLocation() {
    result = getADeclarationEntry().getLocation()
  }

  predicate isSpecialization() {
    exists(FunctionDeclarationEntry2 fde
    | fun_decls(fde,this,_,_,_) and fde.isSpecialization())
  }

  Specifier2 getASpecifier() {
    funspecifiers(this,result)
    or result.hasName(getADeclarationEntry().getASpecifier())
  }

  predicate hasSpecifier(string name) {
    this.getASpecifier().hasName(name)
  }

  Parameter2 getParameter(int n) { params(result,this,n,_) }

  Class2 getDeclaringType() { member(result,_,this) }
  
  Type2 getATemplateArgument() {
    exists(int i | this.getTemplateArgument(i) = result )
  }

  Type2 getTemplateArgument(int index) {
    function_template_argument(this,index,result)
  }

  string toString() { result = "" }
}

class MemberFunction2 extends Function2 {
  MemberFunction2() { member(_,_,this) }
}

class VirtualFunction2 extends MemberFunction2 {

  VirtualFunction2() {
    this.hasSpecifier("virtual") or purefunctions(this)
  }
}

class Location2 extends @location {
  string toString() { result = "" }
}

class FunctionDeclarationEntry2 extends @fun_decl {
  Location2 getLocation() { fun_decls(this,_,_,_,result) }

  predicate isSpecialization() {
    fun_specialized(this)
  }
  string getASpecifier() { fun_decl_specifiers(this,result) }

  string toString() { result = "" }
}

class Constructor2 extends MemberFunction2 {
  Constructor2() { functions(this,_,2) }
}

class Destructor2 extends MemberFunction2 {
  Destructor2() { functions(this,_,3) }
}

class Specifier2 extends @specifier {
  string getName() { specifiers(this,result) }
  predicate hasName(string name) { name = this.getName() }
  string toString() { result = this.getName() }
}

class Variable2 extends @variable {
  Class2 getDeclaringType() { member(result,_,this) }
  predicate hasSpecifier(string name) {
    this.getASpecifier().hasName(name)
  }
  Specifier2 getASpecifier() { varspecifiers(this,result) }
  predicate isStatic() { this.hasSpecifier("static") }
  Type2 getType() { membervariables(this,result,_) }
  string toString() { result = "" }
}

private predicate hasCopySignature(MemberFunction2 f) {
  f.getParameter(0).getType()
    .getUnspecifiedType()                 // resolve typedefs
    .(LValueReferenceType2).getBaseType() // step through lvalue reference type
    .getUnspecifiedType() =              // resolve typedefs, strip const/volatile
  f.getDeclaringType()
}

class Operator2 extends Function2 {
  Operator2() { functions(this,_,5) }
}

class CopyAssignmentOperator2 extends Operator2 {
  CopyAssignmentOperator2() {
    hasName("operator=") and
    (hasCopySignature(this) or
     // Unlike CopyConstructor, this member allows a non-reference
     // parameter.
     getParameter(0).getType().getUnspecifiedType() = getDeclaringType()
    ) and
    not exists(this.getParameter(1)) and
    not exists(getATemplateArgument())
  }
}

class ReferenceType2 extends @derivedtype {
  ReferenceType2() { derivedtypes(this,_,2,_) or derivedtypes(this,_,8,_) }
  Type2 getBaseType() { derivedtypes(this,_,_,result) }
  string toString() { result = "" }
}

class LValueReferenceType2 extends ReferenceType2 {
  LValueReferenceType2() { derivedtypes(this,_,2,_) }
}

/**
 * Holds if `t` is a scalar type, according to the rules specified in
 * C++03 3.9(10):
 *
 *   Arithmetic types (3.9.1), enumeration types, pointer types, and
 *   pointer to member types (3.9.2), and cv-qualified versions of these
 *   types (3.9.3) are collectively called scalar types.
 */
predicate isScalarType(Type2 t) {
  exists (Type2 ut
  | ut = t.getUnspecifiedType()
  | ut instanceof ArithmeticType2 or
    ut instanceof Enum2 or
    ut instanceof FunctionPointerType2 or
    ut instanceof PointerToMemberType2 or
    ut instanceof PointerType2)
}

/**
 * Holds if `c` is an aggregate class, according to the rules specified in
 * C++03 8.5.1(1):
 *
 *   An aggregate [class] is ... a class (clause 9) with no user-declared
 *   constructors (12.1), no private or protected non-static data members
 *   (clause 11), no base classes (clause 10), and no virtual functions
 *   (10.3).
 */
predicate isAggregateClass(Class2 c) {
  not (c instanceof TemplateClass2) and
  not exists (Constructor2 cons
      | cons.getDeclaringType() = c and
        exists (cons.getADeclarationLocation())) and
  not exists (Variable2 v
      | v.getDeclaringType() = c and
        not v.isStatic()
      | v.hasSpecifier("private") or
        v.hasSpecifier("protected")) and
  not exists (c.getABaseClass()) and
  not exists (VirtualFunction2 f
      | f.getDeclaringType() = c)
}

/**
 * Holds if `c` is a POD class, according to the rules specified in
 * C++03 9(4):
 *
 *   A POD-struct is an aggregate class that has no non-static data members
 *   of type non-POD-struct, non-POD-union (or array of such types) or
 *   reference, and has no user-defined copy assignment operator and no
 *   user-defined destructor. Similarly, a POD-union is an aggregate union
 *   that has no non-static data members of type non-POD-struct,
 *   non-POD-union (or array of such types) or reference, and has no
 *   user-defined copy assignment operator and no user-defined destructor.
 *   A POD class is a class that is either a POD-struct or a POD-union.
 */
predicate isPODClass(Class2 c) {
  isAggregateClass(c) and
  not exists (Variable2 v
      | v.getDeclaringType() = c and
        not v.isStatic()
      | not isPODType(v.getType()) or
        exists (ArrayType2 at
        | at = v.getType() and
          not isPODType(at.getBaseType())) or
        v.getType() instanceof ReferenceType2) and
  not exists (CopyAssignmentOperator2 o | o.getDeclaringType() = c) and
  not exists (Destructor2 dest | dest.getDeclaringType() = c)
}

/**
 * Holds if `t` is a POD type, according to the rules specified in
 * C++03 3.9(10):
 *
 *   Scalar types, POD-struct types, POD-union types (clause 9), arrays of
 *   such types and cv-qualified versions of these types (3.9.3) are
 *   collectively called POD types.
 */
predicate isPODType(Type2 t)
{
  exists (Type2 ut
  | ut = t.getUnspecifiedType()
  | isScalarType(ut) or
    isPODClass(ut) or
    exists (ArrayType2 at | at = ut and isPODType(at.getBaseType())) or
    isPODType(ut.getUnspecifiedType()))
}

from Class2 c
where isPODClass(c)
select c
