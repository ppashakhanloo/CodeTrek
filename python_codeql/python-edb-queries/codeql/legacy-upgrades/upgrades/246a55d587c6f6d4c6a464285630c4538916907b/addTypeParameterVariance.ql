
// BEGIN INLINE DEFAULT LIBRARY FRAGMENTS
class Namespace extends @namespace {
	string getName() { namespaces(this,result) }
	
	predicate hasName(string name) { name = this.getName() }
	
	predicate hasQualifiedName(string qualifier, string name) {
		name = this.getName() and
		if exists(Namespace ns | ns = this.getParentNamespace() and not ns.hasName("")) then 
			qualifier = this.getParentNamespace().getQualifiedName()
		else 
			qualifier = ""
	}
	
	string getQualifiedName() {
		exists(string qualifier, string name | this.hasQualifiedName(qualifier, name) and
			if qualifier = "" then 
				result = name
			else 
				result = qualifier + "." + name
		)
	}

	Namespace getParentNamespace() { parent_namespace(this, result) }

	UnboundGenericType getATypeDeclaration() { parent_namespace(result, this) }

	string toString() { result = this.getQualifiedName() }
}

class UnboundGenericType extends @generic {
	UnboundGenericType() { is_generic(this) }
	
	string toString() { result = getName() }
	
	string getName() { types(this,_,result) }

	predicate hasQualifiedName(string qualifier, string name) {	
		name = this.getName() and
		if exists(Namespace ns | ns.getATypeDeclaration() = this and not ns.hasName("")) then
			qualifier = this.getNamespace().getQualifiedName()
		else
			qualifier = ""
	}

	string getQualifiedName() {
		exists(string qualifier, string name | this.hasQualifiedName(qualifier, name) and
			if qualifier = "" then 
				result = name
			else 
				result = qualifier + "." + name
		)
	}

	Namespace getNamespace() {
		result.getATypeDeclaration() = this
	}
	
	TypeParameter getATypeParameter() { 
		result=getTypeParameter(_) 
	}
	
	int getNumberOfTypeParameters() {
		result = count(getATypeParameter())
	}
	
	TypeParameter getTypeParameter(int n) {
		type_parameters(result, n, this)
	}
}

class TypeParameter extends @type_parameter {
	string toString() { types(this,_,result) }
}

class SystemFuncDelegateType extends UnboundGenericType {
	SystemFuncDelegateType() {
		getQualifiedName().regexpMatch("System.Func<,*>")
	}
	
	TypeParameter getAnInputTypeParameter() {
		exists(int i |
			i in [0 .. getNumberOfTypeParameters() - 2] |
			result = getTypeParameter(i)
		)
	}
	
	TypeParameter getResultTypeParameter() {
		result = getTypeParameter(getNumberOfTypeParameters() - 1)
	}
}

SystemFuncDelegateType getSystemFuncDelegateType() { any() }

class SystemActionDelegateType extends UnboundGenericType {
	SystemActionDelegateType() {
		getQualifiedName().regexpMatch("System.Action<,*>")
	}
}

SystemActionDelegateType getSystemActionDelegateType() { any() }

class SystemPredicateDelegateType extends UnboundGenericType {
	SystemPredicateDelegateType() {
		hasQualifiedName("System", "Predicate<>")
		and
		getNumberOfTypeParameters() = 1
	}
}

SystemPredicateDelegateType getSystemPredicateDelegateType() { any() }

class SystemCollectionsGenericIEnumerableTInterface extends UnboundGenericType {
	SystemCollectionsGenericIEnumerableTInterface() {
		hasQualifiedName("System.Collections.Generic", "IEnumerable<>")
		and
		getNumberOfTypeParameters() = 1
	}
}

SystemCollectionsGenericIEnumerableTInterface getSystemCollectionsGenericIEnumerableTInterface() { any() }

// END INLINE DEFAULT LIBRARY FRAGMENTS

predicate isIn(TypeParameter tp) {
	tp = getSystemFuncDelegateType().getAnInputTypeParameter()
	or
	tp = getSystemActionDelegateType().getATypeParameter()
	or
	tp = getSystemPredicateDelegateType().getTypeParameter(0)
}

predicate isOut(TypeParameter tp) {
	tp = getSystemCollectionsGenericIEnumerableTInterface().getTypeParameter(0)
	or
	tp = getSystemFuncDelegateType().getResultTypeParameter()
}

from TypeParameter tp, int index, UnboundGenericType ugt, int variance
where
	type_parameters(tp, index, ugt) and
	if isIn(tp) then variance = 2 else if isOut(tp) then variance = 1 else variance = 0
select tp, index, ugt, variance