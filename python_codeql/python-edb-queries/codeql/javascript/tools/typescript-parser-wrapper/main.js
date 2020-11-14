"use strict";
var __spreadArrays = (this && this.__spreadArrays) || function () {
    for (var s = 0, i = 0, il = arguments.length; i < il; i++) s += arguments[i].length;
    for (var r = Array(s), k = 0, i = 0; i < il; i++)
        for (var a = arguments[i], j = 0, jl = a.length; j < jl; j++, k++)
            r[k] = a[j];
    return r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
var pathlib = require("path");
var readline = require("readline");
var ts = require("./typescript");
var ast_extractor = require("./ast_extractor");
var common_1 = require("./common");
var type_table_1 = require("./type_table");
var virtual_source_root_1 = require("./virtual_source_root");
Error.stackTraceLimit = Infinity;
var State = (function () {
    function State() {
        this.project = null;
        this.typeTable = new type_table_1.TypeTable();
        this.pendingFiles = [];
        this.pendingFileIndex = 0;
        this.pendingResponse = null;
        this.parsedPackageJson = new Map();
        this.packageTypings = new Map();
        this.enclosingPackageJson = new Map();
    }
    return State;
}());
var state = new State();
var reloadMemoryThresholdMb = getEnvironmentVariable("SEMMLE_TYPESCRIPT_MEMORY_THRESHOLD", Number, 1000);
function getPackageJson(file) {
    var cache = state.parsedPackageJson;
    if (cache.has(file))
        return cache.get(file);
    var result = getPackageJsonRaw(file);
    cache.set(file, result);
    return result;
}
function getPackageJsonRaw(file) {
    if (!ts.sys.fileExists(file))
        return undefined;
    try {
        var json = JSON.parse(ts.sys.readFile(file));
        if (typeof json !== 'object')
            return undefined;
        return json;
    }
    catch (e) {
        return undefined;
    }
}
function getPackageTypings(file) {
    var cache = state.packageTypings;
    if (cache.has(file))
        return cache.get(file);
    var result = getPackageTypingsRaw(file);
    cache.set(file, result);
    return result;
}
function getPackageTypingsRaw(packageJsonFile) {
    var json = getPackageJson(packageJsonFile);
    if (json == null)
        return undefined;
    var typings = json.types || json.typings;
    if (typeof typings !== 'string')
        return undefined;
    var absolutePath = pathlib.join(pathlib.dirname(packageJsonFile), typings);
    if (ts.sys.directoryExists(absolutePath)) {
        absolutePath = pathlib.join(absolutePath, 'index.d.ts');
    }
    else if (!absolutePath.endsWith('.ts')) {
        absolutePath += '.d.ts';
    }
    if (!ts.sys.fileExists(absolutePath))
        return undefined;
    return ts.sys.resolvePath(absolutePath);
}
function getEnclosingPackageJson(file) {
    var cache = state.packageTypings;
    if (cache.has(file))
        return cache.get(file);
    var result = getEnclosingPackageJsonRaw(file);
    cache.set(file, result);
    return result;
}
function getEnclosingPackageJsonRaw(file) {
    var packageJson = pathlib.join(file, 'package.json');
    if (ts.sys.fileExists(packageJson)) {
        return packageJson;
    }
    if (pathlib.basename(file) === 'node_modules') {
        return undefined;
    }
    var dirname = pathlib.dirname(file);
    if (dirname.length < file.length) {
        return getEnclosingPackageJson(dirname);
    }
    return undefined;
}
function checkCycle(root) {
    var path = [];
    function visit(obj) {
        if (obj == null || typeof obj !== "object")
            return false;
        if (obj.$cycle_visiting) {
            return true;
        }
        obj.$cycle_visiting = true;
        for (var k in obj) {
            if (!obj.hasOwnProperty(k))
                continue;
            if (+k !== +k && !astPropertySet.has(k))
                continue;
            if (k === "$cycle_visiting")
                continue;
            var cycle = visit(obj[k]);
            if (cycle) {
                path.push(k);
                return true;
            }
        }
        obj.$cycle_visiting = undefined;
        return false;
    }
    visit(root);
    if (path.length > 0) {
        path.reverse();
        console.log(JSON.stringify({ type: "error", message: "Cycle = " + path.join(".") }));
    }
}
var astProperties = [
    "$declarationKind",
    "$declaredSignature",
    "$end",
    "$lineStarts",
    "$overloadIndex",
    "$pos",
    "$resolvedSignature",
    "$symbol",
    "$tokens",
    "$type",
    "argument",
    "argumentExpression",
    "arguments",
    "assertsModifier",
    "asteriskToken",
    "attributes",
    "block",
    "body",
    "caseBlock",
    "catchClause",
    "checkType",
    "children",
    "clauses",
    "closingElement",
    "closingFragment",
    "condition",
    "constraint",
    "constructor",
    "declarationList",
    "declarations",
    "decorators",
    "default",
    "delete",
    "dotDotDotToken",
    "elements",
    "elementType",
    "elementTypes",
    "elseStatement",
    "escapedText",
    "exclamationToken",
    "exportClause",
    "expression",
    "exprName",
    "extendsType",
    "falseType",
    "finallyBlock",
    "flags",
    "head",
    "heritageClauses",
    "importClause",
    "incrementor",
    "indexType",
    "init",
    "initializer",
    "isExportEquals",
    "isTypeOf",
    "isTypeOnly",
    "keywordToken",
    "kind",
    "label",
    "left",
    "literal",
    "members",
    "messageText",
    "modifiers",
    "moduleReference",
    "moduleSpecifier",
    "name",
    "namedBindings",
    "objectType",
    "openingElement",
    "openingFragment",
    "operand",
    "operator",
    "operatorToken",
    "parameterName",
    "parameters",
    "parseDiagnostics",
    "properties",
    "propertyName",
    "qualifier",
    "questionDotToken",
    "questionToken",
    "right",
    "selfClosing",
    "statement",
    "statements",
    "tag",
    "tagName",
    "template",
    "templateSpans",
    "text",
    "thenStatement",
    "token",
    "tokenPos",
    "trueType",
    "tryBlock",
    "type",
    "typeArguments",
    "typeName",
    "typeParameter",
    "typeParameters",
    "types",
    "variableDeclaration",
    "whenFalse",
    "whenTrue",
];
var astMetaProperties = [
    "ast",
    "type",
];
var astPropertySet = new Set(__spreadArrays(astProperties, astMetaProperties));
function stringifyAST(obj) {
    return JSON.stringify(obj, function (k, v) {
        return (+k === +k || astPropertySet.has(k)) ? v : undefined;
    });
}
function extractFile(filename) {
    var ast = getAstForFile(filename);
    return stringifyAST({
        type: "ast",
        ast: ast,
    });
}
function prepareNextFile() {
    if (state.pendingResponse != null)
        return;
    if (state.pendingFileIndex < state.pendingFiles.length) {
        checkMemoryUsage();
        var nextFilename = state.pendingFiles[state.pendingFileIndex];
        state.pendingResponse = extractFile(nextFilename);
    }
}
function handleParseCommand(command, checkPending) {
    if (checkPending === void 0) { checkPending = true; }
    var filename = command.filename;
    var expectedFilename = state.pendingFiles[state.pendingFileIndex];
    if (expectedFilename !== filename && checkPending) {
        throw new Error("File requested out of order. Expected '" + expectedFilename + "' but got '" + filename + "'");
    }
    ++state.pendingFileIndex;
    var response = state.pendingResponse || extractFile(command.filename);
    state.pendingResponse = null;
    process.stdout.write(response + "\n", function () {
        prepareNextFile();
    });
}
function isExtractableSourceFile(ast) {
    return ast.redirectInfo == null;
}
function getAstForFile(filename) {
    if (state.project != null) {
        var ast_1 = state.project.program.getSourceFile(filename);
        if (ast_1 != null && isExtractableSourceFile(ast_1)) {
            ast_extractor.augmentAst(ast_1, ast_1.text, state.project);
            return ast_1;
        }
    }
    var _a = parseSingleFile(filename), ast = _a.ast, code = _a.code;
    ast_extractor.augmentAst(ast, code, null);
    return ast;
}
function parseSingleFile(filename) {
    var code = ts.sys.readFile(filename);
    var compilerHost = {
        fileExists: function () { return true; },
        getCanonicalFileName: function () { return filename; },
        getCurrentDirectory: function () { return ""; },
        getDefaultLibFileName: function () { return "lib.d.ts"; },
        getNewLine: function () { return "\n"; },
        getSourceFile: function () {
            return ts.createSourceFile(filename, code, ts.ScriptTarget.Latest, true);
        },
        readFile: function () { return null; },
        useCaseSensitiveFileNames: function () { return true; },
        writeFile: function () { return null; },
        getDirectories: function () { return []; },
    };
    var compilerOptions = {
        experimentalDecorators: true,
        experimentalAsyncFunctions: true,
        jsx: ts.JsxEmit.Preserve,
        noResolve: true,
    };
    var program = ts.createProgram([filename], compilerOptions, compilerHost);
    var ast = program.getSourceFile(filename);
    return { ast: ast, code: code };
}
var nodeModulesRex = /[/\\]node_modules[/\\]((?:@[\w.-]+[/\\])?\w[\w.-]*)[/\\](.*)/;
function loadTsConfig(command) {
    var tsConfig = ts.readConfigFile(command.tsConfig, ts.sys.readFile);
    var basePath = pathlib.dirname(command.tsConfig);
    var packageEntryPoints = new Map(command.packageEntryPoints);
    var packageJsonFiles = new Map(command.packageJsonFiles);
    var virtualSourceRoot = new virtual_source_root_1.VirtualSourceRoot(command.sourceRoot, command.virtualSourceRoot);
    function redirectNodeModulesPath(path) {
        var nodeModulesMatch = nodeModulesRex.exec(path);
        if (nodeModulesMatch == null)
            return null;
        var packageName = nodeModulesMatch[1];
        var packageJsonFile = packageJsonFiles.get(packageName);
        if (packageJsonFile == null)
            return null;
        var packageDir = pathlib.dirname(packageJsonFile);
        var suffix = nodeModulesMatch[2];
        var finalPath = pathlib.join(packageDir, suffix);
        if (!ts.sys.fileExists(finalPath))
            return null;
        return finalPath;
    }
    var parseConfigHost = {
        useCaseSensitiveFileNames: true,
        readDirectory: function (rootDir, extensions, excludes, includes, depth) {
            var exclusions = excludes == null ? [] : __spreadArrays(excludes);
            if (virtualSourceRoot.virtualSourceRoot != null) {
                exclusions.push(virtualSourceRoot.virtualSourceRoot);
            }
            var originalResults = ts.sys.readDirectory(rootDir, extensions, exclusions, includes, depth);
            var virtualDir = virtualSourceRoot.toVirtualPath(rootDir);
            if (virtualDir == null) {
                return originalResults;
            }
            var virtualExclusions = excludes == null ? [] : __spreadArrays(excludes);
            virtualExclusions.push('**/node_modules/**/*');
            var virtualResults = ts.sys.readDirectory(virtualDir, extensions, virtualExclusions, includes, depth);
            return __spreadArrays(originalResults, virtualResults);
        },
        fileExists: function (path) {
            return ts.sys.fileExists(path)
                || virtualSourceRoot.toVirtualPathIfFileExists(path) != null
                || redirectNodeModulesPath(path) != null;
        },
        readFile: function (path) {
            if (!ts.sys.fileExists(path)) {
                var virtualPath = virtualSourceRoot.toVirtualPathIfFileExists(path);
                if (virtualPath != null)
                    return ts.sys.readFile(virtualPath);
                virtualPath = redirectNodeModulesPath(path);
                if (virtualPath != null)
                    return ts.sys.readFile(virtualPath);
            }
            return ts.sys.readFile(path);
        }
    };
    var config = ts.parseJsonConfigFileContent(tsConfig.config, parseConfigHost, basePath);
    var ownFiles = config.fileNames.map(function (file) { return pathlib.resolve(file); });
    return { config: config, basePath: basePath, packageJsonFiles: packageJsonFiles, packageEntryPoints: packageEntryPoints, virtualSourceRoot: virtualSourceRoot, ownFiles: ownFiles };
}
function handleGetFileListCommand(command) {
    var _a = loadTsConfig(command), config = _a.config, ownFiles = _a.ownFiles;
    console.log(JSON.stringify({
        type: "file-list",
        ownFiles: ownFiles,
    }));
}
function handleOpenProjectCommand(command) {
    var _a = loadTsConfig(command), config = _a.config, packageEntryPoints = _a.packageEntryPoints, virtualSourceRoot = _a.virtualSourceRoot, basePath = _a.basePath, ownFiles = _a.ownFiles;
    var project = new common_1.Project(command.tsConfig, config, state.typeTable, packageEntryPoints, virtualSourceRoot);
    project.load();
    state.project = project;
    var program = project.program;
    var typeChecker = program.getTypeChecker();
    var shouldReportDiagnostics = getEnvironmentVariable("SEMMLE_TYPESCRIPT_REPORT_DIAGNOSTICS", Boolean, false);
    var diagnostics = shouldReportDiagnostics
        ? program.getSemanticDiagnostics().filter(function (d) { return d.category === ts.DiagnosticCategory.Error; })
        : [];
    if (diagnostics.length > 0) {
        console.warn('TypeScript: reported ' + diagnostics.length + ' semantic errors.');
    }
    for (var _i = 0, diagnostics_1 = diagnostics; _i < diagnostics_1.length; _i++) {
        var diagnostic = diagnostics_1[_i];
        var text = diagnostic.messageText;
        if (text && typeof text !== 'string') {
            text = text.messageText;
        }
        var locationStr = '';
        var file = diagnostic.file;
        if (file != null) {
            var _b = file.getLineAndCharacterOfPosition(diagnostic.start), line = _b.line, character = _b.character;
            locationStr = file.fileName + ":" + line + ":" + character;
        }
        console.warn("TypeScript: " + locationStr + " " + text);
    }
    var typeRoots = ts.getEffectiveTypeRoots(config.options, {
        directoryExists: function (path) { return ts.sys.directoryExists(path); },
        getCurrentDirectory: function () { return basePath; },
    });
    for (var _c = 0, _d = typeRoots || []; _c < _d.length; _c++) {
        var typeRoot = _d[_c];
        if (ts.sys.directoryExists(typeRoot)) {
            traverseTypeRoot(typeRoot, "");
        }
        var virtualTypeRoot = virtualSourceRoot.toVirtualPathIfDirectoryExists(typeRoot);
        if (virtualTypeRoot != null) {
            traverseTypeRoot(virtualTypeRoot, "");
        }
    }
    for (var _e = 0, _f = program.getSourceFiles(); _e < _f.length; _e++) {
        var sourceFile = _f[_e];
        addModuleBindingsFromModuleDeclarations(sourceFile);
        addModuleBindingsFromFilePath(sourceFile);
    }
    function joinModulePath(prefix, suffix) {
        if (prefix.length === 0)
            return suffix;
        if (suffix.length === 0)
            return prefix;
        return prefix + "/" + suffix;
    }
    function traverseTypeRoot(filePath, importPrefix) {
        for (var _i = 0, _a = fs.readdirSync(filePath); _i < _a.length; _i++) {
            var child = _a[_i];
            if (child[0] === ".")
                continue;
            var childPath = pathlib.join(filePath, child);
            if (fs.statSync(childPath).isDirectory()) {
                traverseTypeRoot(childPath, joinModulePath(importPrefix, child));
                continue;
            }
            var sourceFile = program.getSourceFile(childPath);
            if (sourceFile == null) {
                continue;
            }
            var importPath = getImportPathFromFileInFolder(importPrefix, child);
            addModuleBindingFromImportPath(sourceFile, importPath);
        }
    }
    function getImportPathFromFileInFolder(folder, baseName) {
        var stem = getStem(baseName);
        return (stem === "index")
            ? folder
            : joinModulePath(folder, stem);
    }
    function addModuleBindingFromImportPath(sourceFile, importPath) {
        var symbol = typeChecker.getSymbolAtLocation(sourceFile);
        if (symbol == null)
            return;
        var canonicalSymbol = getEffectiveExportTarget(symbol);
        var symbolId = state.typeTable.getSymbolId(canonicalSymbol);
        state.typeTable.addModuleMapping(symbolId, importPath);
        if (symbol.globalExports != null) {
            symbol.globalExports.forEach(function (global) {
                state.typeTable.addGlobalMapping(symbolId, global.name);
            });
        }
    }
    function getStem(file) {
        if (file.endsWith(".d.ts")) {
            return pathlib.basename(file, ".d.ts");
        }
        var base = pathlib.basename(file);
        var dot = base.lastIndexOf('.');
        return dot === -1 || dot === 0 ? base : base.substring(0, dot);
    }
    function addModuleBindingsFromFilePath(sourceFile) {
        var fullPath = sourceFile.fileName;
        var index = fullPath.lastIndexOf('/node_modules/');
        if (index === -1)
            return;
        var relativePath = fullPath.substring(index + '/node_modules/'.length);
        if (relativePath.startsWith("@types/"))
            return;
        var packageJsonFile = getEnclosingPackageJson(fullPath);
        if (packageJsonFile != null) {
            var json = getPackageJson(packageJsonFile);
            var typings = getPackageTypings(packageJsonFile);
            if (json != null && typings != null) {
                var name = json.name;
                if (typings === fullPath && typeof name === 'string') {
                    addModuleBindingFromImportPath(sourceFile, name);
                }
                else if (typings != null) {
                    return;
                }
            }
        }
        var _a = pathlib.parse(relativePath), dir = _a.dir, base = _a.base;
        addModuleBindingFromImportPath(sourceFile, getImportPathFromFileInFolder(dir, base));
    }
    function addModuleBindingsFromModuleDeclarations(sourceFile) {
        for (var _i = 0, _a = sourceFile.statements; _i < _a.length; _i++) {
            var stmt = _a[_i];
            if (ts.isModuleDeclaration(stmt) && ts.isStringLiteral(stmt.name)) {
                var symbol = stmt.symbol;
                if (symbol == null)
                    continue;
                symbol = getEffectiveExportTarget(symbol);
                var symbolId = state.typeTable.getSymbolId(symbol);
                var moduleName = stmt.name.text;
                state.typeTable.addModuleMapping(symbolId, moduleName);
            }
        }
    }
    function getEffectiveExportTarget(symbol) {
        if (symbol.exports != null && symbol.exports.has(ts.InternalSymbolName.ExportEquals)) {
            var exportAlias = symbol.exports.get(ts.InternalSymbolName.ExportEquals);
            if (exportAlias.flags & ts.SymbolFlags.Alias) {
                return typeChecker.getAliasedSymbol(exportAlias);
            }
        }
        return symbol;
    }
    var allFiles = program.getSourceFiles().map(function (sf) { return pathlib.resolve(sf.fileName); });
    console.log(JSON.stringify({
        type: "project-opened",
        ownFiles: ownFiles,
        allFiles: allFiles,
    }));
}
function handleCloseProjectCommand(command) {
    if (state.project == null) {
        console.log(JSON.stringify({
            type: "error",
            message: "No project is open",
        }));
        return;
    }
    state.project.unload();
    state.project = null;
    console.log(JSON.stringify({ type: "project-closed" }));
}
function handleGetTypeTableCommand(command) {
    console.log(JSON.stringify({
        type: "type-table",
        typeTable: state.typeTable.getTypeTableJson(),
    }));
}
function handleResetCommand(command) {
    reset();
    console.log(JSON.stringify({
        type: "reset-done",
    }));
}
function handlePrepareFilesCommand(command) {
    state.pendingFiles = command.filenames;
    state.pendingFileIndex = 0;
    state.pendingResponse = null;
    process.stdout.write('{"type":"ok"}\n', function () {
        prepareNextFile();
    });
}
function handleGetMetadataCommand(command) {
    console.log(JSON.stringify({
        type: "metadata",
        syntaxKinds: ts.SyntaxKind,
        nodeFlags: ts.NodeFlags,
    }));
}
function reset() {
    state = new State();
    state.typeTable.restrictedExpansion = getEnvironmentVariable("SEMMLE_TYPESCRIPT_NO_EXPANSION", Boolean, true);
}
function getEnvironmentVariable(name, parse, defaultValue) {
    var value = process.env[name];
    return value != null ? parse(value) : defaultValue;
}
var hasReloadedSinceExceedingThreshold = false;
function checkMemoryUsage() {
    var bytesUsed = process.memoryUsage().heapUsed;
    var megabytesUsed = bytesUsed / 1000000;
    if (!hasReloadedSinceExceedingThreshold && megabytesUsed > reloadMemoryThresholdMb && state.project != null) {
        console.warn('Restarting TypeScript compiler due to memory usage');
        state.project.reload();
        hasReloadedSinceExceedingThreshold = true;
    }
    else if (hasReloadedSinceExceedingThreshold && megabytesUsed < reloadMemoryThresholdMb) {
        hasReloadedSinceExceedingThreshold = false;
    }
}
function runReadLineInterface() {
    reset();
    var rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.on("line", function (line) {
        var req = JSON.parse(line);
        switch (req.command) {
            case "parse":
                handleParseCommand(req);
                break;
            case "open-project":
                handleOpenProjectCommand(req);
                break;
            case "get-own-files":
                handleGetFileListCommand(req);
                break;
            case "close-project":
                handleCloseProjectCommand(req);
                break;
            case "get-type-table":
                handleGetTypeTableCommand(req);
                break;
            case "prepare-files":
                handlePrepareFilesCommand(req);
                break;
            case "reset":
                handleResetCommand(req);
                break;
            case "get-metadata":
                handleGetMetadataCommand(req);
                break;
            case "quit":
                rl.close();
                break;
            default:
                throw new Error("Unknown command " + req.command + ".");
        }
    });
}
if (process.argv.length > 2) {
    var argument = process.argv[2];
    if (argument === "--version") {
        console.log("parser-wrapper with TypeScript " + ts.version);
    }
    else if (pathlib.basename(argument) === "tsconfig.json") {
        handleOpenProjectCommand({
            command: "open-project",
            tsConfig: argument,
            packageEntryPoints: [],
            packageJsonFiles: [],
            sourceRoot: null,
            virtualSourceRoot: null,
        });
        for (var _i = 0, _a = state.project.program.getSourceFiles(); _i < _a.length; _i++) {
            var sf = _a[_i];
            if (pathlib.basename(sf.fileName) === "lib.d.ts")
                continue;
            handleParseCommand({
                command: "parse",
                filename: sf.fileName,
            }, false);
        }
    }
    else if (pathlib.extname(argument) === ".ts" || pathlib.extname(argument) === ".tsx") {
        handleParseCommand({
            command: "parse",
            filename: argument,
        }, false);
    }
    else {
        console.error("Unrecognized file or flag: " + argument);
    }
    process.exit(0);
}
else {
    runReadLineInterface();
}
