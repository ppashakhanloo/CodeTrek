class Commit extends @svnentry {
string toString() { result = "commit" }
}
class File extends @file {
string toString() { result = "file" }
}

from Commit c, File f, int churn
where svnchurn(c,f,churn)
select c, f, churn, 0
