"""Star catalogs, behind one interface.

Each source here answers the same question -- which catalog stars fall in this field -- from a
different backing store: a local binary catalog, a Gaia mirror, a remote cone search. Which one
runs is a config choice, so a deployment with no network still solves.
"""
