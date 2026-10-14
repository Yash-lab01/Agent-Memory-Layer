# Spec 003: Multi-Factor Memory Scoring Engine

## 1. Goal
Implement the multi-factor memory ranking and scoring engine in `src/memlayer/retrieve/scoring.py` for Milestone 3 (Phase 3.2). The engine combines semantic/hybrid similarity, exponential recency decay, and intrinsic importance ratings into a principled, transparent composite score:
$$\text{Score} = a \cdot \text{Similarity} + b \cdot \exp\left(-\frac{\text{age}}{\tau}\right) + c \cdot \text{Importance}$$

---

## 2. Requirements & Mathematical Specifications

### 2.1 Hyperparameters (`ScoringWeights`)
- $a \ge 0.0$: Weight for query relevance / similarity (default: $0.5$).
- $b \ge 0.0$: Weight for recency decay (default: $0.3$).
- $c \ge 0.0$: Weight for intrinsic importance (default: $0.2$).
- $\tau > 0.0$: Exponential decay scale factor in hours (default: $72.0$ hours / 3 days).
- Invariant check: $a, b, c \ge 0.0$ and $\tau > 0.0$.

### 2.2 Age & Recency Decay
- $\text{age} = \max\left(0.0, \frac{\text{now} - \text{memory.created\_at}}{3600}\right)$ (in hours).
- If $\text{memory.created\_at}$ is missing timezone info, treat as UTC.
- Future timestamps ($\text{now} < \text{created\_at}$) are clamped to $\text{age} = 0.0$ (recency component $= 1.0$).
- When $\text{age} = 0.0 \implies \exp(0) = 1.0$.
- When $\text{age} \to \infty \implies \exp(-\infty) = 0.0$.
- If $\tau \to \infty$ or decay is disabled, recency component $= 1.0$.

### 2.3 Importance Component
- Uses `memory.importance` ($0.0 \le \text{importance} \le 1.0$).
- If `importance` is unset or `None`, default to a neutral baseline of $0.5$.

### 2.4 Output Data Contract
Returns `ScoredMemory`:
- `memory`: The original `MemoryItem`.
- `score`: Composite score.
- `similarity`: The similarity score passed in.
- `recency_score`: Unweighted recency factor $\exp(-\text{age}/\tau) \in [0.0, 1.0]$.
- `importance_score`: The importance value utilized $\in [0.0, 1.0]$.

---

## 3. Acceptance Criteria
1. **Mathematical Accuracy:** Verified against analytical calculations for age $0$, age $\tau$, age $2\tau$, and large age.
2. **Boundary Stability:**
   - Handles missing or extreme timestamps without throwing exceptions.
   - Handles missing importance gracefully.
   - Throws `ValueError` on invalid weights ($a, b, c < 0$ or $\tau \le 0$).
3. **Deterministic Reference Time:** Allows passing `now: datetime` explicitly for reproducible offline evaluations and unit tests.
4. **Ranking Helper:** Provides `score_memories(...)` and `rank_memories(...)` to batch score and sort candidate memories.
5. **Quality Gates:** 100% test coverage for scoring logic, passes `ruff`, `mypy`, and `pytest`.
