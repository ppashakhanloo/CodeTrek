import java
import pruning

from Field fieldId, FieldDeclaration fieldDeclId, int pos
where fieldDeclaredIn(fieldId, fieldDeclId, pos)
  and isSourceLocation(fieldId.getLocation())
select fieldId, fieldDeclId, pos