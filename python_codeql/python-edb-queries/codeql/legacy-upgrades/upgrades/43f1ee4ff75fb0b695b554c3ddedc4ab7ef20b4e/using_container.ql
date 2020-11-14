
class Using extends @using { string toString() { none() } }
class Element extends @element { string toString() { none() } }
class LocationDefault extends @location_default { string toString() { none() } }

from Using id, Element element_id, Element container, LocationDefault location
where usings(id, element_id, container, location)
select container, id

