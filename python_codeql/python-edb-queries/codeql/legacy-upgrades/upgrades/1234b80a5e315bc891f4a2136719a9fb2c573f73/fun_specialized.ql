
class FunDecl extends @fun_decl {
    string toString() { none() }
}

class Function extends @function {
    string toString() { none() }
}

class Location extends @location_default {
    string toString() { none() }
}

from FunDecl spec_fd, Function spec_f, Location spec_l
where fun_decls(spec_fd, spec_f, _, _, spec_l)
  and not exists(Function tmpl_f | function_instantiation(spec_f, tmpl_f)
                               and fun_decls(_, tmpl_f, _, _, spec_l))
select spec_fd

