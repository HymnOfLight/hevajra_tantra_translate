"""Statistics over the evidence-graded matrix (synthesis 6).

    twophase   strata, posterior-predictive draws, prevalence (E1/E2), Manski bounds,
               blind -> final revision rates
    contrast   stratified Delta (E4) with its permutation test, TOST, caliper matching and
               the overlap / misclassification diagnostics
    decompose  E3 decomposition classes, O / E / O - E and excess fractions, NOT_ESTIMABLE gate
    power      planning power for Delta and for the over-attribution experiment
    distance   witness distances and UPGMA (descriptive; needs >= 3 witnesses)

Everything is pure and deterministic given a seed; stdlib only.
"""
