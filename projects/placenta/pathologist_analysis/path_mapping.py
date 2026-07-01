"""Mapping of Labelbox annotator name -> whether they are a pathologist.

Used by analysis.py to separate expert (pathologist) annotations from others.
Keys are the exact "Created By" strings in your Labelbox export; values are
1 for a pathologist and 0 otherwise.

Replace the example entries below with your own annotators.
"""

PATHOLOGIST_MAPPING = {
    # "annotator@example.com": 1,   # pathologist
    # "student@example.com": 0,     # non-pathologist
}
