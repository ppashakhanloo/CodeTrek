/**
 * This upgrade script supports the dbscheme change:
 * 
 *  macroinvocations(unique int id: @macroinvocation,
 *            int macro_id: @ppd_define ref,
 *            int location: @location_default ref,
 * +          int kind: int ref);
 *
 * where kind:
 *  1 = macro expansion
 *  2 = other macro reference
 *
 */

class MacroAccess extends @macroinvocation {
	string toString() {result = "macro access"}
}

class Macro extends @ppd_define {
	string toString() {result = "macro"}
}

class LocationDefault extends @location_default {
	string toString() {result = "location default"}
}

from MacroAccess id, Macro macro_id, LocationDefault location
where macroinvocations(id, macro_id, location)
select id, macro_id, location, 1
