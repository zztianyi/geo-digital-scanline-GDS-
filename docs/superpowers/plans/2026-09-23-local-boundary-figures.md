# Local boundary figure review ledger

Spec: C:/Users/222/Downloads/GDS_Local_Boundary_Review_Figure_Codex_Plan.md
Scope: read-only figures and preliminary human-readable observations. No A1/A2/A3/B0/B1, route, P2 or threshold changes; no geometry scoring rerun.
Source decisions: outputs/facetrack_local_preference/20260923_105650. Source observed geometry: existing compact profile cache referenced by outputs/facetrack_voting_phase1/20260923_094440/manifest.json.
Plot inventory checked: locc_local_review_plot.py (paper helpers, exact display clipping, line collection, export); render_facetrack_voting_phase1.py (same-axis source/neighbor galleries); render_selected_cases.py (larger local context); render_facetrack_competition_review.py (all-candidate rows). Extend the existing plotting library with boundary figure adapters, preserve its headless/no-solver contract.
Ruling: Plan's example S IDs are inconsistent with frozen data. The actual 3/5/8 V PF16 islands are R00806_S002/S004/S009. Use their real left and right boundaries, plus S007/S008. D boundaries are raw dissent boundaries inside the same S001, not newly invented subregion boundaries. E boundary is S010/S011.
Ruling: Phase 1 has a winning physical group, not a final assembled winner branch. Thick lines show that group's observed candidate branches only; no fabricated new main spine. Grey context is all original observed section geometry, not a reassembled main track.
Ruling: Main A panels use a fixed shared local window with an overlay row and separate source rows so nearly coincident candidates are not hidden. B shows both original-coordinate overlay and display-only horizontal translation at a unique same-z anchor; no rotation, scaling, smoothing, vertical shift or solver write-back. If no unique common anchor is available, disclose and retain original coordinates.
Ruling: Reuse frozen CC boundaries as markers and frozen A3 scores as explanatory evidence only. No new scoring. Do not presume S007/S008 is a true positive before inspecting curves.
Checks: one artifact/provenance coverage check, no algorithm tests for this read-only plotting change. Visual inspection of all boundary overlay figures is necessary for the requested preliminary observations.

Tasks:
1. Complete: existing plotting library adapters and 12-boundary manifest. No recognition functions imported or changed.
2. Complete: 24 A/B images and 12 C tables generated from 61 cached profiles; case-specific observations after visual review. Output: outputs/local_boundary_review/20260923_boundary/LOCAL_BOUNDARY_REVIEW_REPORT.md.
3. Complete: independent read-only review found no critical/important issues. One artifact/provenance check passed: 12 batches, 24 PNGs, 12 C tables, 70 profile records, 145 frozen A3 rows, 38 report links; all 77 frozen source/cache files unchanged. See outputs/local_boundary_review/20260923_boundary/ARTIFACT_CHECK.json. No algorithm tests or full run.

Font issue: configured Chinese font is absent locally, and the generic fallback inherited the paper font without Chinese glyphs. Stopped the draft export, reused Microsoft YaHei from render_facetrack_voting_phase1 in boundary adapters only, regenerated once; original plotting functions/configuration unchanged.
Final review rulings: three-dimensional true surface identity, final merging and universal correctness of 0.99 were not judged, as this is a frozen local geometry review. They remain explicitly manual/unproven in the report. Report gives conditional recommendations without changing any decision.
Visual findings: 3V/8V islands favor merge review; 5V left boundary weak but right side has a real hook and fragments; large S007/S008 boundary is not a demonstrated positive; D local dissent does not support new subregions; E has actual cached branch-connectivity evidence, provisionally retain. No generic island-size rule applied.
