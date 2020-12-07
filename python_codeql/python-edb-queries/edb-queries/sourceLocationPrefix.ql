import python

from string prefix
where sourceLocationPrefix(prefix)
select prefix.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")