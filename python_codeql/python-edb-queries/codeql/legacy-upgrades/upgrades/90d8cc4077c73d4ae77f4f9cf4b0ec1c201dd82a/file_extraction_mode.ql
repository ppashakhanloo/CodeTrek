class File extends @file {

  /** whether this file may be from source */
  predicate fromSource() { 
    this.getExtension().toLowerCase() = "cs"
  }

  /** the full name of this file */
  string getName() { files(this,result,_,_,_) }

  /** a printable representation of this file */
  string toString() { result = this.getName() }

  /** the URL of this file */
  string getURL() { result = "file://" + this.getName() + ":0:0:0:0" }

  /** the extension of this file */
  string getExtension() { files(this,_,_,result,_) }
}

/**
 * A C# source file.
 */ 
class SourceFile extends File {
  SourceFile() { this.fromSource() }
}

from SourceFile f
select f, 0
