class Locatable extends @locatable {
  int getNumLines() { numlines(this, result, _, _) }
  int getNumCode() { numlines(this, _, result, _) }
  int getNumComment() { numlines(this, _, _, result) }
  string toString() { result = "locatable" }
}

class File extends Locatable, @file {
  override int getNumLines() {
    result = sum(TopLevel tl | this = tl.getFile() | any(int l | numlines(tl, l, _, _)))
  }
  override int getNumCode() {
    result = sum(TopLevel tl | this = tl.getFile() | any(int l | numlines(tl, _, l, _)))
  }
  override int getNumComment() {
    result = sum(TopLevel tl | this = tl.getFile() | any(int l | numlines(tl, _, _, l)))
  }
}

class TopLevel extends Locatable, @toplevel {
  File getFile() {
    exists (@location loc | hasLocation(this, loc) |
      locations_default(loc, result, _, _, _, _)
    )
  }
}

from Locatable l
where l instanceof File or numlines(l, _, _, _)
select l, l.getNumLines(), l.getNumCode(), l.getNumComment()
