"""Triple classification without editing established CI evidence."""


def classify(b, separators, sepsetA, sepsetC, maj_rule):
    """Classify a triple from separating sets and return the evidence for both directions."""
    evidence = {tuple(sorted(s)) for s in separators}
    for known in (sepsetA, sepsetC):
        if known is not None:
            evidence.add(tuple(sorted(known)))
    if not evidence:
        return 3, {"sepsetA": sepsetA, "sepsetC": sepsetC}
    present = sum(b in s for s in evidence)
    fraction = present / len(evidence)
    status = 3
    if (maj_rule and fraction < 0.5) or (not maj_rule and present == 0):
        status = 1
    elif (maj_rule and fraction > 0.5) or (not maj_rule and present == len(evidence)):
        status = 2
    chosen = set(min(evidence, key=lambda s: (len(s), s)))
    return status, {
        "sepsetA": set(sepsetA) if sepsetA is not None else chosen,
        "sepsetC": set(sepsetC) if sepsetC is not None else chosen,
    }
