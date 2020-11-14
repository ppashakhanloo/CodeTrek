/*
 * Convert all type kind -1 into type kind 31.
 * 
 * Case 1: A type is uniquely -1 -> kind = 31
 * Case 2: A type is uniquely defined  -> Copy tuple
 * Case 3: A type is both -1 and defined -> Remove the tuple with -1. 
 * Case 4: A type does not have a unique kind -> kind = 31
 * 
 */

class MyType extends @type { string toString() { result="type" } }

from MyType t, int oldKind, string name,int newKind
where
	types(t, oldKind, name)
and if 
	// The type has a unique kind (apart from -1)
	count(int k | types(t, k, name) and k!=-1) = 1
then 
	newKind != -1 
	and types(t, newKind, name)
else
	newKind = 31
select t, newKind, name
