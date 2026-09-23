# Phase 1 FaceTrack voting execution ledger

Spec: C:/Users/222/Downloads/GDS_Architecture_Refactor_Phase1_FaceTrack_Voting_Plan.md
User: implement Phase 1, full 2800-profile analysis, spatial distribution, representative galleries, performance report.

Ruling: Work on the supplied dirty workspace; add a separate Phase 1 entry and modules. Do not commit, reset, create a checkout missing prior repairs, or alter old route algorithms.
Ruling: The user supplied and authorized the architectural specification. No repeat design approval. This phase stops at immutable region decisions; no junction/route/P2 execution.
Ruling: Latest user explicitly requests full analysis, superseding earlier local-only limits. Retain a one-hour runtime guard; profile/optimize if projection exceeds it.
Ruling: Read exact frozen physical branches and mesh CSR; do not reslice/reload/weld the GLB. Hash the actual source bytes and mesh artifacts for cache provenance.
Ruling: Use global shared-edge mesh components as stable physical FaceTrack identities. A branch may have several members within one physical group. Record local historical T -> physical group correspondence; never compare old local T labels directly.
Ruling: Define spatial coexistence on a fixed 0.5 m U/Z lattice for each original V. Cells contain actual observed segment intervals, not branch bounding boxes. Only cells with >=2 physical groups enter conflict voting. Connected 26-neighbor cells are merged only when their exact competing-group sets agree; a missing V or changed group set stops propagation. Region geometry is the cell mask, not its envelope. Resolution is explicit and reviewable; sub-cell boundary uncertainty remains.
Ruling: Presence is raw observed presence, independent of MBG. MBG gates eligibility, not source identity. This preserves failed candidates in the review table.
Ruling: Through means a contiguous original branch traversal of the conflict mask with observed path both before and after the mask; endpoints inside do not count. CC support means the eligible witness's actual competing intervals lie wholly in its canonical CC; report partial fraction too. Witness selection is evidence-only, never a route edit.
Ruling: Exit continuation is capped at a configured 1 m of the same observed branch, in its low-end to high-end path orientation, and never follows another branch. Comparisons quantize this evidence to 1 mm to avoid meaningless floating-point winners.
Ruling: Per-V vote priority is eligibility, through, full CC, bounded continuation. Region reducer priority is through V count, longest consecutive eligible V run and its s span, CC V count, bounded continuation mean. Exact ties remain ambiguous; no ID tiebreak.
Ruling: The explicit user minimal-check rule overrides repeated TDD/suite runs: write focused behavioral contracts first and run one scoped suite after implementation; actual full analysis also checks conservation, independence, frozen outputs and cache keys.

Pre-flight: A1 outputs immutable branch metrics / compact geometry / cell memberships; A2 consumes cells only; A3 consumes frozen geometry and region masks; B consumes only score rows. Scoring code and policy have a separate cache key so scoring changes do not rebuild geometry.

Tasks:
1. Complete: standalone evidence/voting API and focused tests (8 initial, 10 after targeted fixes).
2. Complete: all-profile A1/A2 indexes, content-addressed caches, parallel A3 and immutable B.
3. Complete: full index-cold run, cache reuse check, two spatial maps and mandatory C05/F, C07, E, D, AUTO_IDENTITY_REVIEW, single-group negative control. Also show C07 upper, eligible tie and no-eligible controls.
4. Complete: reports, measured performance, one fresh code review, necessary fixes. Human physical-identity acceptance remains pending.

Independent reviewer: phase1_review. No Critical issues. Two Important findings fixed: through now requires one contiguous competition interval; in_CC no longer requires full cell coverage. Added two focused regression examples, one allowed targeted rerun: 10 tests passed in 0.009 s. No full test suite/build/commit/push.
Minor limitations retained and disclosed: current source is a fixed 0.05 m complete dataset; nonuniform sampling and all-empty Arrow schemas are not generalized this phase.

Initial full run: outputs/facetrack_voting_phase1/20260923_093358, 70.147 s, 2800 V, 20017 branches, 35 physical components, 963 regions, 25873 score rows. RSS sampling every result dominated A3. Changed instrumentation to every 100 results; one full same-input A3 performance replay measured 31.348 -> 3.862 s (8.116x matched-stage speedup), all score rows and decisions identical.
Final score-policy-v2 run: outputs/facetrack_voting_phase1/20260923_094440, 9.644 s including hash verification/exports. Same geometry index 74301afa346be42814f92eb40e91058be847ed845bd48afcd1cd7127c4406bfa reused; A3 2.931 s; B 0.0779 s. 40 region winners/ambiguity states changed versus the initial implementation. Cache-only source/hash recheck: 1.614 s.
Final output: 344 unique region winners; 594 ties and 25 no-eligible regions. 9048/9186 evaluable neighboring-V votes agree, but 1937/10887 winner-region/V combinations oppose their region winner. This is internal consistency, not accuracy.
Important scientific outcome: E selects PF16 region-wide despite local E_T2/PF19 preference; F local PF16 conflicts with region PF19; C07 and 89.95 share region R00268 but have opposite local preferences. The implementation now exposes this rather than claiming frozen identity implies correct geometry. No case-specific scoring exceptions or route application added.

Atlas QA: corrected first-row title spacing, numeric x-axis formatter on categorical heatmap, single-row legend spacing. Eligible tie R00761 and no-eligible R00008 are explicitly separate. All representative pages list all visible physical sources and branches, even sources outside the page's selected competition mask. Regional heatmaps and complete per-V tables supplement local five-neighbor figures. Read-only rendering preserves frozen SHA256.
Deliverables: ARCHITECTURE_REFACTOR_REPORT.md, PERFORMANCE_REPORT.md, REGION_CATALOG.md, 9 competition case pages + single-source negative page, spatial maps, branch/FaceTrack/scores parquet and fixed region/decision JSON in the final output directory.
