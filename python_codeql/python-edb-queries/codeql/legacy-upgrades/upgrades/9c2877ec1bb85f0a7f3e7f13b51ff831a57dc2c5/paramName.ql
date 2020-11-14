class Parameter extends @param {
  string toString() { result = "parameter" }
}

from Parameter p, string name
where
  exists(int pos | params(p,name,_,pos,_,_) |
    not exists(string other | params(p,other,_,_,_,_) and other != name) or
    name != "p"+pos
  )
select p, name
