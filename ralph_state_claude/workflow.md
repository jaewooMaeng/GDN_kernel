# GDN Decode Optimization Workflow

## Current Phase
**Phase 3 (Performance Optimization)** — Iteration #1 complete

## Baseline & Target
- **Accepted Baseline**: 0.012920 ms (5-run median from Phase 3)
- **Phase 4 Target**: < 0.009 ms
- **Gap**: 0.003920 ms (30% improvement needed)

## History

### Iteration #1: G5 + A3 (2026-04-24)
| Metric | Value | Status |
|--------|-------|--------|
| avg_latency_ms | 0.011415 | ✅ Pass (PASSED all 54 workloads) |
| kernel_duration_us | 33.31 | ✅ Profiled |
| issue_slots_busy_pct | 17.33 | ⚠️ Still low |
| achieved_occupancy_pct | 27.34 | ⚠️ Limited by registers |
| l1_hit_rate_pct | 5.37 | ⚠️ Minimal gain |
| improvement_vs_baseline | +11.6% | ⚠️ Modest but positive |

**Decision**: Proceed to B5 (warp specialization)
**Reason**: __ldg() shows that micro-optimization has limited ROI. Next must address register pressure (56→48-52) and bytes-in-flight through structural changes.

## Candidate Optimizations (Prioritized)

### 🔴 B5: Warp Specialization (Next, High Priority)
- **Goal**: Reduce register pressure, improve occupancy
- **Method**: 1 producer warp + 3 consumer warps; async load overlap
- **Expected**: occupancy +30-50%, latency -0.001~-0.002 ms
- **Risk**: mbarrier sync overhead could increase latency
- **Status**: Pending next iteration

### 🔴 B2: Async Pipeline + Double Buffer (Medium Priority)
- **Goal**: Hide memory latency with compute
- **Method**: Pipeline state loads with compute stages
- **Expected**: latency -0.0005~-0.001 ms
- **Status**: Pending

### 🔴 J1: State Layout Repack (Low Priority)
- **Goal**: Improve L1/L2 hit rates
- **Method**: k-last → v-last or contiguous layout
- **Expected**: latency -0.0005 ms
- **Risk**: API compatibility, layout convert overhead
- **Status**: Hold unless B5 insufficient

## Learned Patterns

| Iteration Range | Category | Outcome |
|-----------------|----------|---------|
| R1-R5 | Cache/Barrier tuning | Failed; overhead > benefit |
| R6-R7 | PTX-level + size | Failed; SASS mismatch |
| R8-R10 | FFMA + async | Failed; ILP still bottleneck |
| **Iter#1** | **__ldg() hint** | **+11.6%; modest but positive** |

**Key insight**: Low issue slots (17.33%) + low occupancy (27.34%) + register limit (56) form a **triple constraint**. Next iteration must address register budget to unlock occupancy.

