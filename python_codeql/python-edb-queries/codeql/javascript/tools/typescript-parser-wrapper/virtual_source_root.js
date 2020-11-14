"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.VirtualSourceRoot = void 0;
var pathlib = require("path");
var ts = require("./typescript");
var VirtualSourceRoot = (function () {
    function VirtualSourceRoot(sourceRoot, virtualSourceRoot) {
        this.sourceRoot = sourceRoot;
        this.virtualSourceRoot = virtualSourceRoot;
    }
    VirtualSourceRoot.translate = function (oldRoot, newRoot, path) {
        if (!oldRoot || !newRoot)
            return null;
        var relative = pathlib.relative(oldRoot, path);
        if (relative.startsWith('..') || pathlib.isAbsolute(relative))
            return null;
        return pathlib.join(newRoot, relative);
    };
    VirtualSourceRoot.prototype.toVirtualPath = function (path) {
        var virtualSourceRoot = this.virtualSourceRoot;
        if (path.startsWith(virtualSourceRoot)) {
            return null;
        }
        return VirtualSourceRoot.translate(this.sourceRoot, virtualSourceRoot, path);
    };
    VirtualSourceRoot.prototype.fromVirtualPath = function (path) {
        return VirtualSourceRoot.translate(this.virtualSourceRoot, this.sourceRoot, path);
    };
    VirtualSourceRoot.prototype.toVirtualPathIfFileExists = function (path) {
        var virtualPath = this.toVirtualPath(path);
        if (virtualPath != null && ts.sys.fileExists(virtualPath)) {
            return virtualPath;
        }
        return null;
    };
    VirtualSourceRoot.prototype.toVirtualPathIfDirectoryExists = function (path) {
        var virtualPath = this.toVirtualPath(path);
        if (virtualPath != null && ts.sys.directoryExists(virtualPath)) {
            return virtualPath;
        }
        return null;
    };
    return VirtualSourceRoot;
}());
exports.VirtualSourceRoot = VirtualSourceRoot;
