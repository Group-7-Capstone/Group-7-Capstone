"""Shared step logic for preprocessing: reading street text from camera tickets."""

import pandas as pd

# A ticket direction code is either a prefix ("NB ...", "N/B ...", "NWB ...") or in
# parentheses ("(E/B)", "(W/     B)" when the fixed-width split falls inside it).
_CODE = r"[NS]?[EW]|[NS]"
DIRECTION_PREFIX = rf"^\s*({_CODE})\s*/?\s*B\b"
DIRECTION_PAREN = rf"\(\s*({_CODE})\s*/?\s*B\s*\)"

# Street name spellings mapped to one form. Applied to text with spaces removed (the
# ticket text splits words at random), so they must be the same on both sides, not
# correct English: longer forms first.
_NAME_FORMS = [
    ("AVENUE", "AV"),
    ("AVE", "AV"),
    ("BOULEVARD", "BV"),
    ("BLVD", "BV"),
    ("STREET", "ST"),
    ("PARKWAY", "PKWY"),
    ("PKY", "PKWY"),
    ("EXPRESSWAY", "EXPY"),
    ("EXPWY", "EXPY"),
    ("PLACE", "PL"),
    ("ROAD", "RD"),
    ("DRIVE", "DR"),
    ("LANE", "LN"),
    ("COURT", "CT"),
    ("TERRACE", "TER"),
    ("HIGHWAY", "HWY"),
    ("TURNPIKE", "TPKE"),
    ("BEACH", "B"),
    ("NORTH", "N"),
    ("SOUTH", "S"),
    ("EAST", "E"),
    ("WEST", "W"),
    ("FIRST", "1"),
    ("SECOND", "2"),
    ("THIRD", "3"),
    ("FOURTH", "4"),
    ("FIFTH", "5"),
    ("SIXTH", "6"),
    ("SEVENTH", "7"),
    ("EIGHTH", "8"),
    ("NINTH", "9"),
    ("TENTH", "10"),
]


# Street types after `_NAME_FORMS`; an ordinal suffix is dropped only before one,
# so "48TH ST" and LION's "48 STREET" both become "48ST".
_STREET_TYPES = "ST|AV|RD|DR|PL|BV|PKWY|LN|CT|TER|EXPY|HWY|TPKE|WAY|LOOP|WALK"


def normalize_street(text: pd.Series) -> pd.Series:
    """Street text in one comparable form: upper case, no spaces or punctuation,
    `_NAME_FORMS` applied and ordinal suffixes dropped ("E 137 TH ST" -> "E137ST")."""
    out = (
        text.str.upper()
        .str.replace(", NEW YORK, NY", "")
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )
    for long, short in _NAME_FORMS:
        out = out.str.replace(long, short)
    return out.str.replace(
        rf"(\d)(?:ST|ND|RD|TH)(?=(?:{_STREET_TYPES}))", r"\1", regex=True
    )
