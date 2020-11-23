

def SetCatalogs(self, catalogs):
    'Set the package info catalogs.\n\n    Args:\n      catalogs: list, catalogs\n    Raises:\n      PlistNotParsedError: the plist was not parsed\n    '
    if (not hasattr(self, '_plist')):
        raise PlistNotParsedError
    self._plist['catalogs'] = catalogs
    catalogs._changed = True
