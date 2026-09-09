# Results and discussion notes

Generated from verified saved predictions. These are descriptive drafting aids, not automatic significance claims.

## Observed method effects
- Highest observed selectivity: SFT, 0.7425 [95% cluster Bayesian-bootstrap interval 0.6129, 0.8771]. This observed ranking does not establish a significant advantage over every other method.
- Highest observed aita_primary: SimPO, 0.0072 [95% cluster Bayesian-bootstrap interval 0.0041, 0.0104]. This observed ranking does not establish a significant advantage over every other method.

## Intrinsic–external disagreement
Positive mean KVS selectivity but negative mean AITA primary gain: CAA-A, CAA-R, SFT. Inspect the strict metric and stance decomposition before attributing the difference to a mechanism.

## Required qualifications
- One seed: no training-seed variance or reproducibility claim. All fitted checkpoints and target values are held fixed.
- KVS measures selective value suppression; AITA measures candidate-probability shifts, not accuracy or general ethical improvement.
- Intervals use Dirichlet(1) cluster weights (implemented as shared exponential weights), with equal observed refined-value aggregation. KVS clusters are source IDs; AITA clusters are normalized post texts shared across values. The two datasets are reweighted independently.
- The uncertainty model treats observed source/post clusters as exchangeable. Text clustering identifies normalized identical posts, not semantic near-duplicates.
- These are Bayesian-bootstrap intervals and descriptive posterior ranking/comparison summaries. They are not frequentist p-values, family-wise simultaneous intervals, or FDR-adjusted tests. This replaces the earlier section 20 calculation; disclose an analysis change if preregistered.
- Missing AITA refined-value cells stay missing; macro means cover observed strata only. Consult coverage and sample counts.
- Leave-one-value-out and prompt/metric sensitivity reuse predictions; they are not additional training ablations or cross-validation.
- Compute is recorded successful-attempt preparation/training time. Retries may be absent; nominal tokens are not measured throughput. Pareto status uses observed means and is descriptive.
- Hyperparameter selection uses KVS validation only. The near-optimum tolerance 0.01 is descriptive and expressed in KVS rating units.
- All-pairs exploration and mechanism correlations are post hoc. Report limitations and avoid selecting only favorable values/methods.

## References
[Rubin (1981), The Bayesian Bootstrap](https://doi.org/10.1214/aos/1176345338). The post-cluster weighting and fixed-stratum metric aggregation are the analysis choices implemented here.

## Missing optional artifacts
None.
