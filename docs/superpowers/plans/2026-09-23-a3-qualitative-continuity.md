# A3 qualitative continuity implementation plan

Spec: C:/Users/222/Downloads/GDS_A3_Qualitative_Continuity_FaceTrack_Audit_Codex_Plan.md
Execution: executing-plans, in the existing user-authorized local checkout. Preserve prior uncommitted boundary plotting changes; no commit, worktree migration, full test suite, A1/A2 rebuild, mesh reads, route or P2 changes.

## Scope and interfaces

1. Extend facetrack_phase1_evidence.py only on the A3 side: independent raw witness evidence -> continuity_classify -> qualitative witness selection -> MBG/continuity/CC/ambiguous decision trace. Keep segment_cells byte-for-byte behavior/source unchanged. POLICY_VERSION changes only A3 score cache, not A1/A2 identity.
2. Add run_facetrack_a3_qualitative_audit.py: require frozen 20260923_094440 A1/A2 cache; ProcessPool per V extracts geometry evidence once and computes default/loose/strict local traces. Reuse unchanged segment_region/reduce_subregion for B0/B1. Export scores, per-candidate traces, old/new votes, independent micro-decision audit, sensitivity and timings.
3. Add render_facetrack_a3_qualitative_audit.py: reuse current plotting helpers; exact raw-coordinate witness overlays, evidence strips, old-island comparisons, elimination tables for R00806/R00268/R00629/R00204/R00132, Chinese audit report answering all 12 questions.
4. Extend tests/test_facetrack_phase1.py with the eight requested cases plus witness/no-candidate/interval invariants. Run it with tests/test_facetrack_local_preference.py once as the minimum combined verification; no unrelated tests. Write tests first; omit the extra red-only test invocation to honor the user's single-check instruction. A targeted retry is allowed only after fixing a diagnosed failure.

## Evidence and initial policy

Old A3: 25,873 candidates / 11,878 (region,V), MBG pass 25,054. Eligible coverage Q01=.6166, Q05=.9167, Q10=1; exit Q25=0/Q50=1; competition arc Q05=.095 m/Q25=2.700 m. These are distributions, not truth labels.
Initial thresholds: GOOD coverage >=.90, meaningful exit >=.25 m; WEAK coverage <=.50 plus exit <=.05 m. Loose .875/.20/.45/.025; strict .925/.30/.55/.075. Fixed .5 m cells provide the structural scale: allow at most one missing cell (count-only tolerance, not a spatial boundary classification); tiny mask re-entry gaps <=one cell diagonal do not imply broken branches; long covered arc >=2 cells can be sufficient even at a natural terminal end. Continuous covered fraction >=.90 also accommodates minor gaps. All scales exposed in policy; no candidate-to-candidate length delta.
GOOD requires robust mask coverage, a sufficiently continuous observed witness and either meaningful exit or a sufficiently long covered arc. WEAK requires both obviously low coverage (more than one missing cell) and negligible exit; all other cases remain UNCERTAIN. Disjoint mask intervals do not by themselves prove a broken physical branch.
All WEAK candidates are rejected; if none remain, no winner. GOOD outranks UNCERTAIN. Equal continuity class proceeds to Boolean CC; equal CC remains ambiguous. Same-class/CC witness branches are represented by stable branch ID, never by exact exit/coverage/arc. No synthetic joining of separate witnesses.
Sensitivity acceptance is a declared engineering screen, not a paper-calibrated constant: <=5% changed A3 votes over all (region,V), <=1% direct non-ambiguous identity flips per loose/strict variant. Also disclose resolved-only denominators and representative-region outcomes. Do not tune thresholds after seeing desired-case labels.

## Audit rulings

- Preserve exact through_region as diagnostic only; it never appears directly in the decision rank.
- A3 no longer uses continuation_quantum_m for comparisons. Keep the field for compatibility with unchanged downstream reducers, and report that B1 retains its previous policy.
- Distinguish all removed legacy exact-exit tie-breaks from the subset whose margin was <=.02 m. Do not call a 1 m difference a centimetre-level difference.
- MICRO_DIFFERENCE_DECISION_COUNT independently checks for an eligible same-class/same-CC peer of any selected candidate; tests must also inject a forbidden winner and detect it.
- New ambiguous results are valid outcomes, not forced merges or failures. If old islands survive due to CC, say so; do not change CC/A1/A2 just to erase them.
- Source geometry and B0/B1 fingerprints are frozen before execution. Timings distinguish default A3/B0/B1 from the additional sensitivity work and report generation.

## Review focus

- Witness selection cannot reintroduce fine-grained ranking before the FaceTrack comparison.
- One missing cell or a tiny interval gap cannot make otherwise good tracks lose to a raw through=True bit.
- All traces agree on final vote, retain all rejected and ambiguous candidates, and explain each elimination.
- Strong-consensus B0 never swallows a newly ambiguous A3 buffer; downstream results are reported separately from A3.
- ProcessPool output and audit totals are deterministic and use (region,V), not duplicated candidate rows, as the slice denominator.

## Execution and verification

- A3 implementation complete: extraction, absolute classes, representative selection without exact ranking, MBG/class/CC/ambiguity trace, independent micro audit. A1 segment_cells source identity unchanged.
- Minimum check: the two specified unittest files ran once, 36/36 passed (22 Phase1 +14 local preference, 0.823 s); pytest unavailable before collection, so standard-library unittest was used without installing dependencies. No subsequent algorithm changes or repeated tests.
- Full A3 replay only: outputs/facetrack_a3_qualitative/20260923_181926. 2800 V tasks, 963 regions, 25873 candidates, 11878 (region,V), 6 workers. Default A3+B0+B1 2.8231208 s; entire entry including hashes/IO/imports 17.3585513 s. Frozen sources and all compact profiles unchanged. Default/loose/strict micro count 0; sensitivity changed 0.505% / 0.589% of comparisons.
- Read-only atlas complete: 19 PNGs, five case reports and full candidate CSVs, total report, trace parquet, evidence distributions, sensitivity and performance manifests. Raw coordinates only; no mesh reads or new recognition. 43 report links resolve. Corrected footer overlap and uninterrupted dash phase across tiny original edges; re-rendered once without re-running scoring.
- Visual spot checks: C07 overview/detail, E transition, F 3V/8V, island comparison and complete evidence strip. Visual labels matched the actual per-V trace; remaining detailed geometry decisions are explicitly for user review.
- Final review: one fresh read-only reviewer a3_final_review inspected the four changed/added code files, tests, spec and artifacts. No important A3 algorithm finding. Important reporting finding (mixing A3 votes with B1 interval assignments) fixed: C07 228V B1 PF19 =227 PF19+1 PF34 A3; R132 middle 345V B1 unresolved =338 AMB+2 PF31+5 PF34 A3. Also corrected 3V attribution (2 CC +1 class) and missing-cell wording. Verified directly against frozen tables; no algorithm edit or test retry needed.
- Review boundary: target surface identity, full manual image acceptance and whether CC should change remain human judgments. CC and B1 policies remain frozen by task scope; the report states the risks and makes no production-route or accuracy claim. No deferred code findings, commits or pushes.

Progress: implementation, authorized replay, report and minimum verification complete. Geometric acceptance remains manual, particularly the 89.95 m coverage-driven change and preserved small CC islands.
