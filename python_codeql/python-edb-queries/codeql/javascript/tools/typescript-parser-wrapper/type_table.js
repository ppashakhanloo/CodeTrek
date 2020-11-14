"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TypeTable = void 0;
var ts = require("./typescript");
function isTypeReference(type) {
    return (type.flags & ts.TypeFlags.Object) !== 0 &&
        (type.objectFlags & ts.ObjectFlags.Reference) !== 0;
}
function isTypeVariable(type) {
    return (type.flags & ts.TypeFlags.TypeVariable) !== 0;
}
function isTypeAlwaysSafeToExpand(type) {
    var flags = type.flags;
    if (flags & ts.TypeFlags.UnionOrIntersection) {
        return false;
    }
    if (flags & ts.TypeFlags.Object) {
        var objectType = type;
        var objectFlags = objectType.objectFlags;
        if (objectFlags & (ts.ObjectFlags.Reference | ts.ObjectFlags.Mapped)) {
            return false;
        }
    }
    return true;
}
function getEnclosingTypeOfThisType(type) {
    var bound = type.getConstraint();
    if (bound == null)
        return null;
    var target = bound.target;
    if (target == null)
        return null;
    return (target.thisType === type) ? target : null;
}
var typeDefinitionSymbols = ts.SymbolFlags.Class | ts.SymbolFlags.Interface |
    ts.SymbolFlags.TypeAlias | ts.SymbolFlags.EnumMember | ts.SymbolFlags.Enum;
function isTypeDefinitionSymbol(symbol) {
    return (symbol.flags & typeDefinitionSymbols) !== 0;
}
function getEnclosingBlock(node) {
    while (true) {
        if (node == null)
            return null;
        if (ts.isSourceFile(node) || ts.isFunctionLike(node) || ts.isBlock(node) || ts.isModuleBlock(node))
            return node;
        node = node.parent;
    }
}
var typeofSymbols = ts.SymbolFlags.Class | ts.SymbolFlags.Namespace |
    ts.SymbolFlags.Module | ts.SymbolFlags.Enum | ts.SymbolFlags.EnumMember;
function isTypeofCandidateSymbol(symbol) {
    return (symbol.flags & typeofSymbols) !== 0;
}
var signatureKinds = [ts.SignatureKind.Call, ts.SignatureKind.Construct];
var TypeTable = (function () {
    function TypeTable() {
        this.typeIds = new Map();
        this.typeToStringValues = [];
        this.typeChecker = null;
        this.arbitraryAstNode = null;
        this.symbolIds = new Map();
        this.fileIds = new Map();
        this.signatureIds = new Map();
        this.signatureToStringValues = [];
        this.propertyLookups = {
            baseTypes: [],
            names: [],
            propertyTypes: [],
        };
        this.typeAliases = {
            aliasTypes: [],
            underlyingTypes: [],
        };
        this.signatureMappings = {
            baseTypes: [],
            kinds: [],
            indices: [],
            signatures: []
        };
        this.numberIndexTypes = {
            baseTypes: [],
            propertyTypes: [],
        };
        this.stringIndexTypes = {
            baseTypes: [],
            propertyTypes: [],
        };
        this.buildTypeWorklist = [];
        this.expansiveTypes = new Map();
        this.moduleMappings = {
            symbols: [],
            names: [],
        };
        this.globalMappings = {
            symbols: [],
            names: [],
        };
        this.baseTypes = {
            symbols: [],
            baseTypeSymbols: [],
        };
        this.selfTypes = {
            symbols: [],
            selfTypes: [],
        };
        this.isInShallowTypeContext = false;
        this.typeExtractionState = [];
        this.typeRecursionDepth = 0;
        this.restrictedExpansion = false;
    }
    TypeTable.prototype.setProgram = function (program, virtualSourceRoot) {
        this.typeChecker = program.getTypeChecker();
        this.arbitraryAstNode = program.getSourceFiles()[0];
        this.virtualSourceRoot = virtualSourceRoot;
    };
    TypeTable.prototype.releaseProgram = function () {
        this.typeChecker = null;
        this.arbitraryAstNode = null;
    };
    TypeTable.prototype.buildType = function (type, unfoldAlias) {
        this.isInShallowTypeContext = false;
        var id = this.getId(type, unfoldAlias);
        this.iterateBuildTypeWorklist();
        if (id == null)
            return null;
        return id;
    };
    TypeTable.prototype.getId = function (type, unfoldAlias) {
        if (this.typeRecursionDepth > 100) {
            return null;
        }
        if ((type.flags & ts.TypeFlags.StringLiteral) && type.value.length > 30) {
            type = this.typeChecker.getBaseTypeOfLiteralType(type);
        }
        ++this.typeRecursionDepth;
        var content = this.getTypeString(type, unfoldAlias);
        --this.typeRecursionDepth;
        if (content == null)
            return null;
        var id = this.typeIds.get(content);
        if (id == null) {
            var stringValue = this.stringifyType(type, unfoldAlias);
            if (stringValue == null) {
                return null;
            }
            id = this.typeIds.size;
            this.typeIds.set(content, id);
            this.typeToStringValues.push(stringValue);
            this.buildTypeWorklist.push([type, id, unfoldAlias]);
            this.typeExtractionState.push(this.isInShallowTypeContext ? 0 : 2);
            if (content.startsWith("reference;") && !(isTypeReference(type) && type.target !== type)) {
                this.selfTypes.symbols.push(this.getSymbolId(type.aliasSymbol || type.symbol));
                this.selfTypes.selfTypes.push(id);
            }
        }
        else if (!this.isInShallowTypeContext) {
            var state = this.typeExtractionState[id];
            if (state === 0) {
                this.typeExtractionState[id] = 2;
            }
            else if (state === 1) {
                this.typeExtractionState[id] = 2;
                this.buildTypeWorklist.push([type, id, unfoldAlias]);
            }
        }
        return id;
    };
    TypeTable.prototype.stringifyType = function (type, unfoldAlias) {
        var formatFlags = unfoldAlias
            ? ts.TypeFormatFlags.InTypeAlias
            : ts.TypeFormatFlags.UseAliasDefinedOutsideCurrentScope;
        var toStringValue;
        try {
            toStringValue = this.typeChecker.typeToString(type, undefined, formatFlags);
        }
        catch (e) {
            console.warn("Recovered from a compiler crash while stringifying a type. Discarding the type.");
            console.warn(e.stack);
            return null;
        }
        if (toStringValue.length > 50) {
            return toStringValue.substring(0, 47) + "...";
        }
        else {
            return toStringValue;
        }
    };
    TypeTable.prototype.stringifySignature = function (signature, kind) {
        var toStringValue;
        try {
            toStringValue =
                this.typeChecker.signatureToString(signature, signature.declaration, ts.TypeFormatFlags.UseAliasDefinedOutsideCurrentScope, kind);
        }
        catch (e) {
            console.warn("Recovered from a compiler crash while stringifying a signature. Discarding the signature.");
            console.warn(e.stack);
            return null;
        }
        if (toStringValue.length > 70) {
            return toStringValue.substring(0, 69) + "...";
        }
        else {
            return toStringValue;
        }
    };
    TypeTable.prototype.getTypeString = function (type, unfoldAlias) {
        if (!unfoldAlias && type.aliasSymbol != null) {
            var tag = "reference;" + this.getSymbolId(type.aliasSymbol);
            return type.aliasTypeArguments == null
                ? tag
                : this.makeTypeStringVector(tag, type.aliasTypeArguments);
        }
        var flags = type.flags;
        var objectFlags = (flags & ts.TypeFlags.Object) && type.objectFlags;
        var symbol = type.symbol;
        if (symbol != null) {
            if (isTypeReference(type)) {
                var tag = "reference;" + this.getSymbolId(symbol);
                return this.makeTypeStringVectorFromTypeReferenceArguments(tag, type);
            }
            if (flags & ts.TypeFlags.TypeVariable) {
                var enclosingType = getEnclosingTypeOfThisType(type);
                if (enclosingType != null) {
                    return "this;" + this.getId(enclosingType, false);
                }
                else if (symbol.parent == null) {
                    return "lextypevar;" + symbol.name;
                }
                else {
                    return "typevar;" + this.getSymbolId(symbol);
                }
            }
            if ((objectFlags & ts.ObjectFlags.Anonymous) && isTypeofCandidateSymbol(symbol)) {
                return "typeof;" + this.getSymbolId(symbol);
            }
            if (isTypeDefinitionSymbol(symbol)) {
                return "reference;" + this.getSymbolId(type.symbol);
            }
        }
        if (flags === ts.TypeFlags.Any) {
            return "any";
        }
        if (flags === ts.TypeFlags.String) {
            return "string";
        }
        if (flags === ts.TypeFlags.Number) {
            return "number";
        }
        if (flags === ts.TypeFlags.Void) {
            return "void";
        }
        if (flags === ts.TypeFlags.Never) {
            return "never";
        }
        if (flags === ts.TypeFlags.BigInt) {
            return "bigint";
        }
        if (flags & ts.TypeFlags.Null) {
            return "null";
        }
        if (flags & ts.TypeFlags.Undefined) {
            return "undefined";
        }
        if (flags === ts.TypeFlags.ESSymbol) {
            return "plainsymbol";
        }
        if (flags & ts.TypeFlags.Unknown) {
            return "unknown";
        }
        if (flags === ts.TypeFlags.UniqueESSymbol) {
            return "uniquesymbol;" + this.getSymbolId(type.symbol);
        }
        if (flags === ts.TypeFlags.NonPrimitive && type.intrinsicName === "object") {
            return "objectkeyword";
        }
        if (flags === ts.TypeFlags.BooleanLiteral) {
            return type.intrinsicName;
        }
        if (flags & ts.TypeFlags.NumberLiteral) {
            return "numlit;" + type.value;
        }
        if (flags & ts.TypeFlags.StringLiteral) {
            return "strlit;" + type.value;
        }
        if (flags & ts.TypeFlags.BigIntLiteral) {
            var literalType = type;
            var value = literalType.value;
            return "bigintlit;" + (value.negative ? "-" : "") + value.base10Value;
        }
        if (flags & ts.TypeFlags.Union) {
            var unionType = type;
            if (unionType.types.length === 0) {
                return null;
            }
            return this.makeTypeStringVector("union", unionType.types);
        }
        if (flags & ts.TypeFlags.Intersection) {
            var intersectionType = type;
            if (intersectionType.types.length === 0) {
                return null;
            }
            return this.makeTypeStringVector("intersection", intersectionType.types);
        }
        if (isTypeReference(type) && (type.target.objectFlags & ts.ObjectFlags.Tuple)) {
            var tupleReference = type;
            var tupleType = tupleReference.target;
            var minLength = tupleType.minLength != null
                ? tupleType.minLength
                : this.typeChecker.getTypeArguments(tupleReference).length;
            var hasRestElement = tupleType.hasRestElement ? 't' : 'f';
            var prefix = "tuple;" + minLength + ";" + hasRestElement;
            return this.makeTypeStringVectorFromTypeReferenceArguments(prefix, type);
        }
        if (objectFlags & ts.ObjectFlags.Anonymous) {
            return this.makeStructuralTypeVector("object;", type);
        }
        return null;
    };
    TypeTable.prototype.getSymbolId = function (symbol) {
        if (symbol.flags & ts.SymbolFlags.Alias) {
            symbol = this.typeChecker.getAliasedSymbol(symbol);
        }
        var id = symbol.$id;
        if (id != null)
            return id;
        var content = this.getSymbolString(symbol);
        id = this.symbolIds.get(content);
        if (id != null) {
            return symbol.$id = id;
        }
        if (id == null) {
            id = this.symbolIds.size;
            this.symbolIds.set(content, id);
            symbol.$id = id;
            if (this.isGlobalSymbol(symbol)) {
                this.addGlobalMapping(id, symbol.name);
            }
            this.extractSymbolBaseTypes(symbol, id);
        }
        return id;
    };
    TypeTable.prototype.isGlobalSymbol = function (symbol) {
        var parent = symbol.parent;
        if (parent != null) {
            if (parent.escapedName === ts.InternalSymbolName.Global) {
                return true;
            }
            return false;
        }
        if (symbol.declarations == null || symbol.declarations.length === 0)
            return false;
        var declaration = symbol.declarations[0];
        var block = getEnclosingBlock(declaration);
        if (ts.isSourceFile(block) && !this.isModuleSourceFile(block)) {
            return true;
        }
        return false;
    };
    TypeTable.prototype.isModuleSourceFile = function (file) {
        return this.typeChecker.getSymbolAtLocation(file) != null;
    };
    TypeTable.prototype.getSymbolString = function (symbol) {
        var parent = symbol.parent;
        if (parent == null || parent.escapedName === ts.InternalSymbolName.Global) {
            return "root;" + this.getSymbolDeclarationString(symbol) + ";;" + this.rewriteSymbolName(symbol);
        }
        else if (parent.exports != null && parent.exports.get(symbol.escapedName) === symbol) {
            return "member;;" + this.getSymbolId(parent) + ";" + this.rewriteSymbolName(symbol);
        }
        else {
            return "other;" + this.getSymbolDeclarationString(symbol) + ";" + this.getSymbolId(parent) + ";" + this.rewriteSymbolName(symbol);
        }
    };
    TypeTable.prototype.rewriteSymbolName = function (symbol) {
        var _a = this.virtualSourceRoot, virtualSourceRoot = _a.virtualSourceRoot, sourceRoot = _a.sourceRoot;
        var name = symbol.name;
        if (virtualSourceRoot == null || sourceRoot == null)
            return name;
        return name.replace(virtualSourceRoot, sourceRoot);
    };
    TypeTable.prototype.getSymbolDeclarationString = function (symbol) {
        if (symbol.declarations == null || symbol.declarations.length === 0) {
            return "";
        }
        var decl = symbol.declarations[0];
        if (ts.isSourceFile(decl))
            return "";
        return this.getFileId(decl.getSourceFile().fileName) + ":" + decl.pos;
    };
    TypeTable.prototype.getFileId = function (fileName) {
        var id = this.fileIds.get(fileName);
        if (id == null) {
            id = this.fileIds.size;
            this.fileIds.set(fileName, id);
        }
        return id;
    };
    TypeTable.prototype.makeTypeStringVectorFromTypeReferenceArguments = function (tag, type) {
        var target = type.target;
        var typeArguments = this.typeChecker.getTypeArguments(type);
        if (typeArguments == null)
            return tag;
        if (target.typeParameters != null) {
            return this.makeTypeStringVector(tag, typeArguments, target.typeParameters.length);
        }
        else {
            return this.makeTypeStringVector(tag, typeArguments);
        }
    };
    TypeTable.prototype.makeTypeStringVector = function (tag, types, length) {
        if (length === void 0) { length = types.length; }
        var hash = tag;
        for (var i = 0; i < length; ++i) {
            var id = this.getId(types[i], false);
            if (id == null)
                return null;
            hash += ";" + id;
        }
        return hash;
    };
    TypeTable.prototype.tryGetTypeOfSymbol = function (symbol) {
        try {
            return this.typeChecker.getTypeOfSymbolAtLocation(symbol, this.arbitraryAstNode);
        }
        catch (e) {
            console.warn("Could not compute type of '" + this.typeChecker.symbolToString(symbol) + "'");
            return null;
        }
    };
    TypeTable.prototype.makeStructuralTypeVector = function (tag, type) {
        var hash = tag;
        for (var _i = 0, _a = type.getProperties(); _i < _a.length; _i++) {
            var property = _a[_i];
            var propertyType = this.tryGetTypeOfSymbol(property);
            if (propertyType == null)
                return null;
            var propertyTypeId = this.getId(propertyType, false);
            if (propertyTypeId == null)
                return null;
            hash += ";p" + this.getSymbolId(property) + ';' + propertyTypeId;
        }
        for (var _b = 0, signatureKinds_1 = signatureKinds; _b < signatureKinds_1.length; _b++) {
            var kind = signatureKinds_1[_b];
            for (var _c = 0, _d = this.typeChecker.getSignaturesOfType(type, kind); _c < _d.length; _c++) {
                var signature = _d[_c];
                var id = this.getSignatureId(kind, signature);
                if (id == null)
                    return null;
                hash += ";c" + id;
            }
        }
        var indexType = type.getStringIndexType();
        if (indexType != null) {
            var indexTypeId = this.getId(indexType, false);
            if (indexTypeId == null)
                return null;
            hash += ";s" + indexTypeId;
        }
        indexType = type.getNumberIndexType();
        if (indexType != null) {
            var indexTypeId = this.getId(indexType, false);
            if (indexTypeId == null)
                return null;
            hash += ";i" + indexTypeId;
        }
        return hash;
    };
    TypeTable.prototype.addModuleMapping = function (symbolId, moduleName) {
        this.moduleMappings.symbols.push(symbolId);
        this.moduleMappings.names.push(moduleName);
    };
    TypeTable.prototype.addGlobalMapping = function (symbolId, globalName) {
        this.globalMappings.symbols.push(symbolId);
        this.globalMappings.names.push(globalName);
    };
    TypeTable.prototype.getTypeTableJson = function () {
        return {
            typeStrings: Array.from(this.typeIds.keys()),
            typeToStringValues: this.typeToStringValues,
            propertyLookups: this.propertyLookups,
            typeAliases: this.typeAliases,
            symbolStrings: Array.from(this.symbolIds.keys()),
            moduleMappings: this.moduleMappings,
            globalMappings: this.globalMappings,
            signatureStrings: Array.from(this.signatureIds.keys()),
            signatureMappings: this.signatureMappings,
            signatureToStringValues: this.signatureToStringValues,
            numberIndexTypes: this.numberIndexTypes,
            stringIndexTypes: this.stringIndexTypes,
            baseTypes: this.baseTypes,
            selfTypes: this.selfTypes,
        };
    };
    TypeTable.prototype.iterateBuildTypeWorklist = function () {
        var worklist = this.buildTypeWorklist;
        var typeExtractionState = this.typeExtractionState;
        while (worklist.length > 0) {
            var _a = worklist.pop(), type = _a[0], id = _a[1], unfoldAlias = _a[2];
            var isShallowContext = typeExtractionState[id] === 0;
            if (isShallowContext && !isTypeAlwaysSafeToExpand(type)) {
                typeExtractionState[id] = 1;
            }
            else if (type.aliasSymbol != null && !unfoldAlias) {
                typeExtractionState[id] = 3;
                var underlyingTypeId = this.getId(type, true);
                if (underlyingTypeId != null) {
                    this.typeAliases.aliasTypes.push(id);
                    this.typeAliases.underlyingTypes.push(underlyingTypeId);
                }
            }
            else {
                typeExtractionState[id] = 3;
                this.isInShallowTypeContext = isShallowContext || this.isExpansiveTypeReference(type);
                this.extractProperties(type, id);
                this.extractSignatures(type, id);
                this.extractIndexers(type, id);
            }
        }
        this.isInShallowTypeContext = false;
    };
    TypeTable.prototype.getPropertiesToExtract = function (type) {
        if (this.getSelfType(type) === type) {
            var thenSymbol = this.typeChecker.getPropertyOfType(type, "then");
            if (thenSymbol != null) {
                return [thenSymbol];
            }
        }
        return null;
    };
    TypeTable.prototype.extractProperties = function (type, id) {
        var props = this.getPropertiesToExtract(type);
        if (props == null)
            return;
        for (var _i = 0, props_1 = props; _i < props_1.length; _i++) {
            var symbol = props_1[_i];
            var propertyType = this.tryGetTypeOfSymbol(symbol);
            if (propertyType == null)
                continue;
            var propertyTypeId = this.getId(propertyType, false);
            if (propertyTypeId == null)
                continue;
            this.propertyLookups.baseTypes.push(id);
            this.propertyLookups.names.push(symbol.name);
            this.propertyLookups.propertyTypes.push(propertyTypeId);
        }
    };
    TypeTable.prototype.getSignatureId = function (kind, signature) {
        var content = this.getSignatureString(kind, signature);
        if (content == null) {
            return null;
        }
        var id = this.signatureIds.get(content);
        if (id == null) {
            var stringValue = this.stringifySignature(signature, kind);
            if (stringValue == null) {
                return null;
            }
            id = this.signatureIds.size;
            this.signatureIds.set(content, id);
            this.signatureToStringValues.push(stringValue);
        }
        return id;
    };
    TypeTable.prototype.getSignatureString = function (kind, signature) {
        var parameters = signature.getParameters();
        var numberOfTypeParameters = signature.typeParameters == null
            ? 0
            : signature.typeParameters.length;
        var requiredParameters = parameters.length;
        for (var i = 0; i < parameters.length; ++i) {
            if (parameters[i].flags & ts.SymbolFlags.Optional) {
                requiredParameters = i;
                break;
            }
        }
        var hasRestParam = (signature.flags & 1) !== 0;
        var restParameterTag = '';
        if (hasRestParam) {
            if (requiredParameters === parameters.length) {
                requiredParameters = parameters.length - 1;
            }
            if (parameters.length === 0)
                return null;
            var restParameter = parameters[parameters.length - 1];
            var restParameterType = this.tryGetTypeOfSymbol(restParameter);
            if (restParameterType == null)
                return null;
            var restParameterTypeId = this.getId(restParameterType, false);
            if (restParameterTypeId == null)
                return null;
            restParameterTag = '' + restParameterTypeId;
        }
        var returnTypeId = this.getId(signature.getReturnType(), false);
        if (returnTypeId == null) {
            return null;
        }
        var tag = kind + ";" + numberOfTypeParameters + ";" + requiredParameters + ";" + restParameterTag + ";" + returnTypeId;
        for (var _i = 0, _a = signature.typeParameters || []; _i < _a.length; _i++) {
            var typeParameter = _a[_i];
            tag += ";" + typeParameter.symbol.name;
            var constraint = typeParameter.getConstraint();
            var constraintId = void 0;
            if (constraint == null || (constraintId = this.getId(constraint, false)) == null) {
                tag += ";";
            }
            else {
                tag += ";" + constraintId;
            }
        }
        for (var paramIndex = 0; paramIndex < parameters.length; ++paramIndex) {
            var parameter = parameters[paramIndex];
            var parameterType = this.tryGetTypeOfSymbol(parameter);
            if (parameterType == null) {
                return null;
            }
            var isRestParameter = hasRestParam && (paramIndex === parameters.length - 1);
            if (isRestParameter) {
                if (!isTypeReference(parameterType))
                    return null;
                var typeArguments = parameterType.typeArguments;
                if (typeArguments == null || typeArguments.length === 0)
                    return null;
                parameterType = typeArguments[0];
            }
            var parameterTypeId = this.getId(parameterType, false);
            if (parameterTypeId == null) {
                return null;
            }
            tag += ';' + parameter.name + ';' + parameterTypeId;
        }
        return tag;
    };
    TypeTable.prototype.extractSignatures = function (type, id) {
        this.extractSignatureList(type, id, ts.SignatureKind.Call, type.getCallSignatures());
        this.extractSignatureList(type, id, ts.SignatureKind.Construct, type.getConstructSignatures());
    };
    TypeTable.prototype.extractSignatureList = function (type, id, kind, list) {
        var index = -1;
        for (var _i = 0, list_1 = list; _i < list_1.length; _i++) {
            var signature = list_1[_i];
            ++index;
            var signatureId = this.getSignatureId(kind, signature);
            if (signatureId == null)
                continue;
            this.signatureMappings.baseTypes.push(id);
            this.signatureMappings.kinds.push(kind);
            this.signatureMappings.indices.push(index);
            this.signatureMappings.signatures.push(signatureId);
        }
    };
    TypeTable.prototype.extractIndexers = function (type, id) {
        this.extractIndexer(id, type.getStringIndexType(), this.stringIndexTypes);
        this.extractIndexer(id, type.getNumberIndexType(), this.numberIndexTypes);
    };
    TypeTable.prototype.extractIndexer = function (baseType, indexType, table) {
        if (indexType == null)
            return;
        var indexTypeId = this.getId(indexType, false);
        if (indexTypeId == null)
            return;
        table.baseTypes.push(baseType);
        table.propertyTypes.push(indexTypeId);
    };
    TypeTable.prototype.extractSymbolBaseTypes = function (symbol, symbolId) {
        for (var _i = 0, _a = symbol.declarations || []; _i < _a.length; _i++) {
            var decl = _a[_i];
            if (ts.isClassLike(decl) || ts.isInterfaceDeclaration(decl)) {
                for (var _b = 0, _c = decl.heritageClauses || []; _b < _c.length; _b++) {
                    var heritage = _c[_b];
                    for (var _d = 0, _e = heritage.types; _d < _e.length; _d++) {
                        var typeExpr = _e[_d];
                        var superType = this.typeChecker.getTypeFromTypeNode(typeExpr);
                        if (superType == null)
                            continue;
                        var baseTypeSymbol = superType.symbol;
                        if (baseTypeSymbol == null)
                            continue;
                        var baseId = this.getSymbolId(baseTypeSymbol);
                        this.baseTypes.symbols.push(symbolId);
                        this.baseTypes.baseTypeSymbols.push(baseId);
                    }
                }
            }
        }
    };
    TypeTable.prototype.getSelfType = function (type) {
        if (isTypeReference(type) && this.typeChecker.getTypeArguments(type).length > 0) {
            return type.target;
        }
        return null;
    };
    TypeTable.prototype.isExpansiveTypeReference = function (type) {
        if (this.restrictedExpansion) {
            return true;
        }
        var selfType = this.getSelfType(type);
        if (selfType != null) {
            this.checkExpansiveness(selfType);
            var id = this.getId(selfType, false);
            return this.expansiveTypes.get(id);
        }
        return false;
    };
    TypeTable.prototype.checkExpansiveness = function (type) {
        var indexTable = new Map();
        var lowlinkTable = new Map();
        var indexCounter = 0;
        var stack = [];
        var expansionDepthTable = new Map();
        var typeTable = this;
        search(type, 0);
        function search(type, expansionDepth) {
            var id = typeTable.getId(type, false);
            if (id == null)
                return null;
            var index = indexTable.get(id);
            if (index != null) {
                var initialExpansionDepth = expansionDepthTable.get(id);
                if (initialExpansionDepth == null) {
                    return null;
                }
                if (expansionDepth > initialExpansionDepth) {
                    typeTable.expansiveTypes.set(id, true);
                }
                return index;
            }
            var previousResult = typeTable.expansiveTypes.get(id);
            if (previousResult != null) {
                return null;
            }
            index = ++indexCounter;
            indexTable.set(id, index);
            lowlinkTable.set(id, index);
            expansionDepthTable.set(id, expansionDepth);
            var indexOnStack = stack.length;
            stack.push(id);
            for (var _i = 0, _a = type.getProperties(); _i < _a.length; _i++) {
                var symbol = _a[_i];
                var propertyType = this.tryGetTypeOfSymbol(symbol);
                if (propertyType == null)
                    continue;
                traverseType(propertyType);
            }
            if (lowlinkTable.get(id) === index) {
                var isExpansive = false;
                for (var i = indexOnStack; i < stack.length; ++i) {
                    var memberId = stack[i];
                    if (typeTable.expansiveTypes.get(memberId) === true) {
                        isExpansive = true;
                        break;
                    }
                }
                for (var i = indexOnStack; i < stack.length; ++i) {
                    var memberId = stack[i];
                    typeTable.expansiveTypes.set(memberId, isExpansive);
                    expansionDepthTable.set(memberId, null);
                }
                stack.length = indexOnStack;
            }
            return lowlinkTable.get(id);
            function traverseType(type) {
                if (isTypeVariable(type))
                    return 1;
                var depth = 0;
                typeTable.forEachChildType(type, function (child) {
                    depth = Math.max(depth, traverseType(child));
                });
                if (depth === 0) {
                    return 0;
                }
                var selfType = typeTable.getSelfType(type);
                if (selfType != null) {
                    visitEdge(selfType, (depth === 1) ? 0 : 1);
                }
                return 2;
            }
            function visitEdge(successor, weight) {
                var result = search(successor, expansionDepth + weight);
                if (result == null)
                    return;
                lowlinkTable.set(id, Math.min(lowlinkTable.get(id), result));
            }
        }
    };
    TypeTable.prototype.forEachChildType = function (type, callback) {
        if (isTypeReference(type)) {
            var typeArguments = this.typeChecker.getTypeArguments(type);
            if (typeArguments != null) {
                typeArguments.forEach(callback);
            }
        }
        else if (type.flags & ts.TypeFlags.UnionOrIntersection) {
            type.types.forEach(callback);
        }
        else if (type.flags & ts.TypeFlags.Object) {
            var objectType = type;
            var objectFlags = objectType.objectFlags;
            if (objectFlags & ts.ObjectFlags.Anonymous) {
                for (var _i = 0, _a = type.getProperties(); _i < _a.length; _i++) {
                    var symbol = _a[_i];
                    var propertyType = this.tryGetTypeOfSymbol(symbol);
                    if (propertyType == null)
                        continue;
                    callback(propertyType);
                }
                for (var _b = 0, _c = type.getCallSignatures(); _b < _c.length; _b++) {
                    var signature = _c[_b];
                    this.forEachChildTypeOfSignature(signature, callback);
                }
                for (var _d = 0, _e = type.getConstructSignatures(); _d < _e.length; _d++) {
                    var signature = _e[_d];
                    this.forEachChildTypeOfSignature(signature, callback);
                }
                var stringIndexType = type.getStringIndexType();
                if (stringIndexType != null) {
                    callback(stringIndexType);
                }
                var numberIndexType = type.getNumberIndexType();
                if (numberIndexType != null) {
                    callback(numberIndexType);
                }
            }
        }
    };
    TypeTable.prototype.forEachChildTypeOfSignature = function (signature, callback) {
        callback(signature.getReturnType());
        for (var _i = 0, _a = signature.getParameters(); _i < _a.length; _i++) {
            var parameter = _a[_i];
            var paramType = this.tryGetTypeOfSymbol(parameter);
            if (paramType == null)
                continue;
            callback(paramType);
        }
        var typeParameters = signature.getTypeParameters();
        if (typeParameters != null) {
            for (var _b = 0, typeParameters_1 = typeParameters; _b < typeParameters_1.length; _b++) {
                var typeParameter = typeParameters_1[_b];
                var constraint = typeParameter.getConstraint();
                if (constraint == null)
                    continue;
                callback(constraint);
            }
        }
    };
    return TypeTable;
}());
exports.TypeTable = TypeTable;
