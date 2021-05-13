import python

class ExprOrStmt extends ExprOrStmt_ {}

class StrParent extends StrParent_ {}

class VariableParent extends VariableParent_ {}

class BoolParent extends BoolParent_ {}

class MyDictItemList extends DictItemList {}

class MyDictItemListParent extends DictItemListParent {}

class MyDictItem extends DictItem {}

class LocationParent extends LocationParent_ {}

class SourceLine extends @sourceline {
    string toString() { result = "SourceLine" }
}

