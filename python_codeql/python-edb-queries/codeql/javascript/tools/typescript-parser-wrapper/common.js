"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Project = void 0;
var ts = require("./typescript");
var pathlib = require("path");
var packageNameRex = /^(?:@[\w.-]+[/\\]+)?\w[\w.-]*(?=[/\\]|$)/;
var extensions = ['.ts', '.tsx', '.d.ts', '.js', '.jsx'];
function getPackageName(importString) {
    var packageNameMatch = packageNameRex.exec(importString);
    if (packageNameMatch == null)
        return null;
    var packageName = packageNameMatch[0];
    if (packageName.charAt(0) === '@') {
        packageName = packageName.replace(/[/\\]+/g, '/');
    }
    return packageName;
}
var Project = (function () {
    function Project(tsConfig, config, typeTable, packageEntryPoints, virtualSourceRoot) {
        this.tsConfig = tsConfig;
        this.config = config;
        this.typeTable = typeTable;
        this.packageEntryPoints = packageEntryPoints;
        this.virtualSourceRoot = virtualSourceRoot;
        this.program = null;
        this.resolveModuleNames = this.resolveModuleNames.bind(this);
        this.resolutionCache = ts.createModuleResolutionCache(pathlib.dirname(tsConfig), ts.sys.realpath, config.options);
        var host = ts.createCompilerHost(config.options, true);
        host.resolveModuleNames = this.resolveModuleNames;
        host.trace = undefined;
        this.host = host;
    }
    Project.prototype.unload = function () {
        this.typeTable.releaseProgram();
        this.program = null;
    };
    Project.prototype.load = function () {
        var _a = this, config = _a.config, host = _a.host;
        this.program = ts.createProgram(config.fileNames, config.options, host);
        this.typeTable.setProgram(this.program, this.virtualSourceRoot);
    };
    Project.prototype.reload = function () {
        this.unload();
        this.load();
    };
    Project.prototype.resolveModuleNames = function (moduleNames, containingFile, reusedNames, redirectedReference, options) {
        var _this = this;
        var oppositePath = this.virtualSourceRoot.toVirtualPath(containingFile) ||
            this.virtualSourceRoot.fromVirtualPath(containingFile);
        var _a = this, host = _a.host, resolutionCache = _a.resolutionCache;
        return moduleNames.map(function (moduleName) {
            var redirected = _this.redirectModuleName(moduleName, containingFile, options);
            if (redirected != null)
                return redirected;
            if (oppositePath != null) {
                redirected = ts.resolveModuleName(moduleName, oppositePath, options, host, resolutionCache).resolvedModule;
                if (redirected != null)
                    return redirected;
            }
            return ts.resolveModuleName(moduleName, containingFile, options, host, resolutionCache).resolvedModule;
        });
    };
    Project.prototype.redirectModuleName = function (moduleName, containingFile, options) {
        var packageName = getPackageName(moduleName);
        if (packageName == null)
            return null;
        var packageEntryPoint = this.packageEntryPoints.get(packageName);
        if (packageEntryPoint == null)
            return null;
        if (moduleName === packageName) {
            return { resolvedFileName: packageEntryPoint, isExternalLibraryImport: true };
        }
        var suffix = moduleName.substring(packageName.length);
        var packageDir = pathlib.dirname(packageEntryPoint);
        var joinedPath = pathlib.join(packageDir, suffix);
        if (ts.sys.directoryExists(joinedPath)) {
            joinedPath = pathlib.join(joinedPath, 'index');
        }
        for (var _i = 0, extensions_1 = extensions; _i < extensions_1.length; _i++) {
            var ext = extensions_1[_i];
            var candidate = joinedPath.endsWith(ext) ? joinedPath : (joinedPath + ext);
            if (ts.sys.fileExists(candidate)) {
                return { resolvedFileName: candidate, isExternalLibraryImport: true };
            }
        }
        return null;
    };
    return Project;
}());
exports.Project = Project;
