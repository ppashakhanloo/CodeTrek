import java

from Field fieldId, FieldDeclaration fieldDeclId, int pos
where fieldDeclaredIn(fieldId, fieldDeclId, pos)
select fieldId, fieldDeclId, pos