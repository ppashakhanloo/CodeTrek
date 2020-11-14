class Type extends @type {
  string toString() { none() }
}

from Type array, string name, Type elemType, int dim, Type componentType
where
  arrays(array, name, elemType, dim) and (
    // For one-dimensional arrays, the component type is just the element type.
    dim = 1 and componentType = elemType or

    dim > 1 and (
      // Try to find an array type with the same element type and one dimension less
      arrays(componentType, _, elemType, dim - 1) or

      // If that type is not populated, then fall back on the element type.
      // Using the element type is not correct, but it's better than leaving
      // such types with no component type at all.
      not arrays(_, _, elemType, dim - 1) and componentType = elemType))
select array, name, elemType, dim, componentType
