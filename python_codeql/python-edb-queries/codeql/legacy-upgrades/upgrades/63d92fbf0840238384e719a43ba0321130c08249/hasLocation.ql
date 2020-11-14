class Locatable extends @locatable {
  string toString() { none() }
}

class Location extends @location {
  string toString() { none() }
}

// Remove function type call signatures from hasLocation

from Locatable locatable, Location location
where hasLocation(locatable, location)
  and not exists (@functiontypeexpr fun | properties(locatable, fun, _, _, _))
select locatable, location
