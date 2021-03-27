import python

from CommentBlock id,string text,Location location
where py_comments(id,text,location)
select id,
       text.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       location