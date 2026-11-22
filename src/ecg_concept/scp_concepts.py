"""PTB-XL SCP form/rhythm statements as supervised CONCEPT targets (report-derived annotations, not independently validated ground truth).

Label policy (declared before any training):
* Candidates = statements flagged `form` or `rhythm` in scp_statements.csv that are NOT also flagged `diagnostic` (NDT, NST_, DIG, LNGQT are form AND diagnostic -> excluded).
* PRESENT  : the statement code is a key of the record's `scp_codes`, whatever its likelihood (likelihood 0 = "unavailable", never "negative"; likelihood is never read here).
* Absence is NOT automatically negative. Two explicit policies, both reported:
    P1 (primary, conservative): a concept is NEGATIVE only if the record carries at least one OTHER statement of the same category (form or rhythm; any of that category's
        non-diagnostic-overlap statements, retained or not) - i.e. the category was documented but this statement was not asserted. If the record carries no statement of the
        category, the concept is UNKNOWN (masked out of the loss). Presence of the diagnostic-overlap statements is ignored for this decision.
    P2 (sensitivity): every absent statement is negative (the usual PTB-XL benchmark convention; an assumption, not a fact).
Targets are int8: 1 present, 0 negative, -1 unknown.
"""
import numpy as np

POLICIES = ("P1", "P2")


def split_statements(statements):
    """statements: dict from ptbxl.read_statements. Returns (form_only, rhythm_only, diag_overlap) as sorted code lists."""
    form = {c for c, s in statements.items() if s["form"]}
    rhythm = {c for c, s in statements.items() if s["rhythm"]}
    diag = {c for c, s in statements.items() if s["diagnostic"]}
    overlap = sorted((form | rhythm) & diag)
    return sorted(form - diag), sorted(rhythm - diag), overlap


def concept_targets(records, vocabulary, category_of, category_members, policy):
    """records: PtbxlRecord list. vocabulary: ordered concept codes. category_of: code -> 'form'|'rhythm'.
    category_members: {'form': [...all non-overlap form codes...], 'rhythm': [...]}. Returns int8 (N, C) in {1, 0, -1}."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    Y = np.zeros((len(records), len(vocabulary)), np.int8)
    members = {k: set(v) for k, v in category_members.items()}
    for i, r in enumerate(records):
        keys = set(r.scp_codes)
        annotated = {cat: bool(keys & m) for cat, m in members.items()}
        for j, c in enumerate(vocabulary):
            if c in keys:
                Y[i, j] = 1
            elif policy == "P1" and not annotated[category_of[c]]:
                Y[i, j] = -1
    return Y
