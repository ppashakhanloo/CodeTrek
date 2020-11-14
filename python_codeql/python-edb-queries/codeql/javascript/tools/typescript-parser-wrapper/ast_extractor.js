"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.augmentAst = void 0;
var ts = require("./typescript");
function hasOwnProperty(o, p) {
    return o && Object.prototype.hasOwnProperty.call(o, p);
}
var SyntaxKind = [];
for (var p in ts.SyntaxKind) {
    if (!hasOwnProperty(ts.SyntaxKind, p)) {
        continue;
    }
    if (+p === +p) {
        continue;
    }
    if (p.substring(0, 5) === "First" || p.substring(0, 4) === "Last") {
        continue;
    }
    SyntaxKind[ts.SyntaxKind[p]] = p;
}
var skipWhiteSpace = /(?:\s|\/\/.*|\/\*[^]*?\*\/)*/g;
function forEachNode(ast, callback) {
    function visit(node) {
        ts.forEachChild(node, visit);
        callback(node);
    }
    visit(ast);
}
function tryGetTypeOfNode(typeChecker, node) {
    try {
        return typeChecker.getTypeAtLocation(node);
    }
    catch (e) {
        var sourceFile = node.getSourceFile();
        var _a = sourceFile.getLineAndCharacterOfPosition(node.pos), line = _a.line, character = _a.character;
        console.warn("Could not compute type of " + ts.SyntaxKind[node.kind] + " at " + sourceFile.fileName + ":" + (line + 1) + ":" + (character + 1));
        return null;
    }
}
function augmentAst(ast, code, project) {
    ast.$lineStarts = ast.getLineStarts();
    function augmentPos(pos, shouldSkipWhitespace) {
        if (shouldSkipWhitespace) {
            skipWhiteSpace.lastIndex = pos;
            pos += skipWhiteSpace.exec(code)[0].length;
        }
        return pos;
    }
    var reScanEvents = [];
    var reScanEventPos = [];
    var scanner = ts.createScanner(ts.ScriptTarget.ES2015, false, 1, code);
    var reScanSlashToken = scanner.reScanSlashToken.bind(scanner);
    var reScanTemplateToken = scanner.reScanTemplateToken.bind(scanner);
    var reScanGreaterToken = scanner.reScanGreaterToken.bind(scanner);
    if (!ast.parseDiagnostics || ast.parseDiagnostics.length === 0) {
        forEachNode(ast, function (node) {
            if (ts.isRegularExpressionLiteral(node)) {
                reScanEventPos.push(node.getStart(ast, false));
                reScanEvents.push(reScanSlashToken);
            }
            if (ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) {
                reScanEventPos.push(node.getStart(ast, false));
                reScanEvents.push(reScanTemplateToken);
            }
            if (ts.isBinaryExpression(node)) {
                var operator = node.operatorToken;
                switch (operator.kind) {
                    case ts.SyntaxKind.GreaterThanEqualsToken:
                    case ts.SyntaxKind.GreaterThanGreaterThanEqualsToken:
                    case ts.SyntaxKind.GreaterThanGreaterThanGreaterThanEqualsToken:
                    case ts.SyntaxKind.GreaterThanGreaterThanGreaterThanToken:
                    case ts.SyntaxKind.GreaterThanGreaterThanToken:
                        reScanEventPos.push(operator.getStart(ast, false));
                        reScanEvents.push(reScanGreaterToken);
                        break;
                }
            }
        });
    }
    reScanEventPos.push(Infinity);
    ast.$tokens = [];
    var rescanEventIndex = 0;
    var nextRescanPosition = reScanEventPos[0];
    var tk;
    do {
        tk = scanner.scan();
        if (scanner.getTokenPos() === nextRescanPosition) {
            var callback = reScanEvents[rescanEventIndex];
            callback();
            ++rescanEventIndex;
            nextRescanPosition = reScanEventPos[rescanEventIndex];
        }
        ast.$tokens.push({
            kind: tk,
            tokenPos: augmentPos(scanner.getTokenPos()),
            text: scanner.getTokenText(),
        });
    } while (tk !== ts.SyntaxKind.EndOfFileToken);
    if (ast.parseDiagnostics) {
        ast.parseDiagnostics.forEach(function (d) {
            delete d.file;
            d.$pos = augmentPos(d.start);
        });
    }
    var typeChecker = project && project.program.getTypeChecker();
    var typeTable = project && project.typeTable;
    if (typeTable != null) {
        var symbol = typeChecker.getSymbolAtLocation(ast);
        if (symbol != null) {
            ast.$symbol = typeTable.getSymbolId(symbol);
        }
    }
    var insideConditionalTypes = 0;
    visitAstNode(ast);
    function visitAstNode(node) {
        if (node.kind === ts.SyntaxKind.ConditionalType) {
            ++insideConditionalTypes;
        }
        ts.forEachChild(node, visitAstNode);
        if (node.kind === ts.SyntaxKind.ConditionalType) {
            --insideConditionalTypes;
        }
        if ("pos" in node) {
            node.$pos = augmentPos(node.pos, true);
        }
        if ("end" in node) {
            node.$end = augmentPos(node.end);
        }
        if (ts.isVariableDeclarationList(node)) {
            var tz = ts;
            if (typeof tz.isLet === "function" && tz.isLet(node) || (node.flags & ts.NodeFlags.Let)) {
                node.$declarationKind = "let";
            }
            else if (typeof tz.isConst === "function" && tz.isConst(node) || (node.flags & ts.NodeFlags.Const)) {
                node.$declarationKind = "const";
            }
            else {
                node.$declarationKind = "var";
            }
        }
        if (typeChecker != null && insideConditionalTypes === 0) {
            if (isTypedNode(node)) {
                var contextualType = isContextuallyTypedNode(node)
                    ? typeChecker.getContextualType(node)
                    : null;
                var type = contextualType || tryGetTypeOfNode(typeChecker, node);
                if (type != null) {
                    var parent = node.parent;
                    var unfoldAlias = ts.isTypeAliasDeclaration(parent) && node === parent.type;
                    var id = typeTable.buildType(type, unfoldAlias);
                    if (id != null) {
                        node.$type = id;
                    }
                }
                if (ts.isCallOrNewExpression(node)) {
                    var kind = ts.isCallExpression(node) ? ts.SignatureKind.Call : ts.SignatureKind.Construct;
                    var resolvedSignature = typeChecker.getResolvedSignature(node);
                    if (resolvedSignature != null) {
                        var resolvedId = typeTable.getSignatureId(kind, resolvedSignature);
                        if (resolvedId != null) {
                            node.$resolvedSignature = resolvedId;
                        }
                        var declaration = resolvedSignature.declaration;
                        if (declaration != null) {
                            var calleeType = typeChecker.getTypeAtLocation(node.expression);
                            if (calleeType != null && declaration != null) {
                                var calleeSignatures = typeChecker.getSignaturesOfType(calleeType, kind);
                                for (var i = 0; i < calleeSignatures.length; ++i) {
                                    if (calleeSignatures[i].declaration === declaration) {
                                        node.$overloadIndex = i;
                                        break;
                                    }
                                }
                            }
                            var name = declaration.name;
                            var symbol = name && typeChecker.getSymbolAtLocation(name);
                            if (symbol != null) {
                                node.$symbol = typeTable.getSymbolId(symbol);
                            }
                        }
                    }
                }
            }
            var symbolNode = isNamedNodeWithSymbol(node) ? node.name :
                ts.isImportDeclaration(node) ? node.moduleSpecifier :
                    ts.isExternalModuleReference(node) ? node.expression :
                        null;
            if (symbolNode != null) {
                var symbol = typeChecker.getSymbolAtLocation(symbolNode);
                if (symbol != null) {
                    node.$symbol = typeTable.getSymbolId(symbol);
                }
            }
            if (ts.isTypeReferenceNode(node)) {
                var namePart = node.typeName;
                while (ts.isQualifiedName(namePart)) {
                    var symbol_1 = typeChecker.getSymbolAtLocation(namePart.right);
                    if (symbol_1 != null) {
                        namePart.$symbol = typeTable.getSymbolId(symbol_1);
                    }
                    namePart = namePart.left;
                }
                var symbol = typeChecker.getSymbolAtLocation(namePart);
                if (symbol != null) {
                    namePart.$symbol = typeTable.getSymbolId(symbol);
                }
            }
            if (ts.isFunctionLike(node)) {
                var signature = typeChecker.getSignatureFromDeclaration(node);
                if (signature != null) {
                    var kind = ts.isConstructSignatureDeclaration(node) || ts.isConstructorDeclaration(node)
                        ? ts.SignatureKind.Construct : ts.SignatureKind.Call;
                    var id = typeTable.getSignatureId(kind, signature);
                    if (id != null) {
                        node.$declaredSignature = id;
                    }
                }
            }
        }
    }
}
exports.augmentAst = augmentAst;
function isNamedNodeWithSymbol(node) {
    switch (node.kind) {
        case ts.SyntaxKind.ClassDeclaration:
        case ts.SyntaxKind.InterfaceDeclaration:
        case ts.SyntaxKind.TypeAliasDeclaration:
        case ts.SyntaxKind.EnumDeclaration:
        case ts.SyntaxKind.EnumMember:
        case ts.SyntaxKind.ModuleDeclaration:
        case ts.SyntaxKind.FunctionDeclaration:
        case ts.SyntaxKind.MethodDeclaration:
        case ts.SyntaxKind.MethodSignature:
            return true;
    }
    return false;
}
function isTypedNode(node) {
    switch (node.kind) {
        case ts.SyntaxKind.ArrayLiteralExpression:
        case ts.SyntaxKind.ArrowFunction:
        case ts.SyntaxKind.AsExpression:
        case ts.SyntaxKind.AwaitExpression:
        case ts.SyntaxKind.BinaryExpression:
        case ts.SyntaxKind.CallExpression:
        case ts.SyntaxKind.ClassExpression:
        case ts.SyntaxKind.ClassDeclaration:
        case ts.SyntaxKind.CommaListExpression:
        case ts.SyntaxKind.ConditionalExpression:
        case ts.SyntaxKind.Constructor:
        case ts.SyntaxKind.DeleteExpression:
        case ts.SyntaxKind.ElementAccessExpression:
        case ts.SyntaxKind.ExpressionStatement:
        case ts.SyntaxKind.ExpressionWithTypeArguments:
        case ts.SyntaxKind.FalseKeyword:
        case ts.SyntaxKind.FunctionDeclaration:
        case ts.SyntaxKind.FunctionExpression:
        case ts.SyntaxKind.GetAccessor:
        case ts.SyntaxKind.Identifier:
        case ts.SyntaxKind.IndexSignature:
        case ts.SyntaxKind.JsxExpression:
        case ts.SyntaxKind.LiteralType:
        case ts.SyntaxKind.MethodDeclaration:
        case ts.SyntaxKind.MethodSignature:
        case ts.SyntaxKind.NewExpression:
        case ts.SyntaxKind.NonNullExpression:
        case ts.SyntaxKind.NoSubstitutionTemplateLiteral:
        case ts.SyntaxKind.NumericLiteral:
        case ts.SyntaxKind.ObjectKeyword:
        case ts.SyntaxKind.ObjectLiteralExpression:
        case ts.SyntaxKind.OmittedExpression:
        case ts.SyntaxKind.ParenthesizedExpression:
        case ts.SyntaxKind.PartiallyEmittedExpression:
        case ts.SyntaxKind.PostfixUnaryExpression:
        case ts.SyntaxKind.PrefixUnaryExpression:
        case ts.SyntaxKind.PropertyAccessExpression:
        case ts.SyntaxKind.RegularExpressionLiteral:
        case ts.SyntaxKind.SetAccessor:
        case ts.SyntaxKind.StringLiteral:
        case ts.SyntaxKind.TaggedTemplateExpression:
        case ts.SyntaxKind.TemplateExpression:
        case ts.SyntaxKind.TemplateHead:
        case ts.SyntaxKind.TemplateMiddle:
        case ts.SyntaxKind.TemplateSpan:
        case ts.SyntaxKind.TemplateTail:
        case ts.SyntaxKind.TrueKeyword:
        case ts.SyntaxKind.TypeAssertionExpression:
        case ts.SyntaxKind.TypeLiteral:
        case ts.SyntaxKind.TypeOfExpression:
        case ts.SyntaxKind.VoidExpression:
        case ts.SyntaxKind.YieldExpression:
            return true;
        default:
            return ts.isTypeNode(node);
    }
}
function isContextuallyTypedNode(node) {
    var kind = node.kind;
    return kind === ts.SyntaxKind.ArrayLiteralExpression || kind === ts.SyntaxKind.ObjectLiteralExpression;
}
