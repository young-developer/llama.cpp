# N-gram Mod Ideas for Increasing Acceptance Rate

## Implemented

### 13. Reject-Then-Learn ✅

When a prediction is rejected, the correct token is known from the extended prompt. Store the rejected n-gram context in `accept()`, then in the next `draft_one()` reconstruct the correct n-gram (context + correct token) and add it before making new predictions.

**Status**: Implemented in `common/speculative.cpp`.

---

## Not Yet Implemented

### 1. Adaptive N-gram Size

Instead of fixed `n_match`, use multiple n-gram sizes. Shorter n-grams hit more often (less specific), longer n-grams are more accurate when they match.

**Approach**: Keep 2-3 hash tables: n=16, n=24, n=32. Try longest first, fall back to shorter. Shorter n-grams have higher hit rate for uncommon patterns.

**Effort**: Medium — add parallel tables, try in order.

---

### 2. N-gram Confidence Ratio

Currently frequency is +1/-1. An n-gram with freq=5 could be 6 right / 1 wrong (86% accuracy) or 1006 right / 1001 wrong (50% accuracy). Track the **acceptance ratio** per entry.

**Approach**: Store both `accept_count` and `reject_count` (or derive from signed freq + total lookups). Evict entries with ratio < 50%, not just freq < -20.

**Effort**: Low — add a second counter per entry, compare ratio in `prune_evict_negative`.

---

### 3. Exponential Decay on Frequency

Old predictions are less relevant than recent ones. A n-gram that was right 100 rounds ago but wrong the last 5 times should be penalized more.

**Approach**: On each accept cycle, halve all frequencies before applying +1/-1:

```cpp
// Before inc/dec:
for (each entry) freq[i] /= 2;  // decay

// Then normal inc/dec:
mod.inc(contexts[i]);  // +1
mod.dec(contexts[i]);  // -1
```

**Effort**: Low — add a decay pass in `accept()`.

---

### 4. Weighted Increment by Position

The first accepted token in a draft is more valuable than the last. The first token's n-gram context is the most reliable (closest to the prompt). Later tokens are predictions built on predictions.

**Approach**: Weight the increment by position:

```cpp
// Position 0: +3, position 1: +2, position 2: +1
for (i = 0; i < n_accepted; ++i) {
    weight = max(1, 3 - i);
    for (w = 0; w < weight; ++w) mod.inc(contexts[i]);
}

// Position 0: -1, position 1: -2, position 2: -3
for (i = n_accepted; i < contexts.size(); ++i) {
    weight = min(3, i - n_accepted + 1);
    for (w = 0; w < weight; ++w) mod.dec(contexts[i]);
}
```

**Effort**: Low — adjust the inc/dec loop in `accept()`.

---

### 5. Dynamic N-gram Size Adjustment

If acceptance rate is consistently low, the n-gram size might be too long (too specific, too many misses). If acceptance is high, try longer n-grams for more aggressive drafting.

**Approach**: Track rolling acceptance rate. Adjust `n_match` at runtime:

```
acceptance > 80% for 10 rounds → increase n_match by 4
acceptance < 30% for 10 rounds → decrease n_match by 4
clamp n_match to [8, 48]
```

**Effort**: Medium — need to rebuild the hash table when n changes, or maintain multiple tables.

---

### 6. Multi-token N-gram Prediction

Currently: predict 1 token at a time, chain them. If token 3 is wrong, tokens 4+ are garbage.

**Approach**: Store n-grams that predict 2-3 tokens at once. The context is `n` tokens, the value is `(token_n+1, token_n+2, token_n+3)`. This reduces error propagation.

**Effort**: High — change the entry type from `int32_t` to a small array, update all methods.

---

### 7. N-gram Cache Warm-up

Cold start: the cache is built from source code but doesn't reflect the model's actual behavior.

**Approach**: After loading the model, run a warm-up phase: feed known prompts, let the model generate, and record which n-gram predictions were accepted vs rejected. This pre-trains the frequency tracking before serving real requests.

**Effort**: Medium — add a warm-up step in server initialization.

---

### 8. Eviction by Acceptance Ratio, Not Just Frequency

An entry with freq=-20 might have been looked up 1000 times (980 right, 1000 wrong → 50% accuracy). An entry with freq=-5 but only 7 total lookups (2 right, 5 wrong → 29% accuracy) is worse.

**Approach**: Track `total_lookups` per entry. Evict by `accept_ratio = (freq + total) / (2 * total)` < threshold:

```cpp
// In get(): increment total_lookups[i]
// In prune:
if ((double)(freq[i] + total[i]) / (2.0 * total[i]) < 0.4) {
    // evict — less than 40% acceptance rate
}
```

**Effort**: Medium — add `total_lookups` array, update in `get()` and `accept()`.

---

### 9. Domain-Specific Sub-tables

Code has domains: C++ syntax, Python syntax, HTML, SQL, etc. An n-gram for C++ templates is irrelevant for SQL queries.

**Approach**: Partition the hash table by domain. Detect domain from prompt (heuristic or classifier), only query the relevant sub-table.

**Effort**: High — need domain detection, multiple sub-tables.

---

### 10. Speculative Draft Chaining

Use ngram-mod for the first few tokens, then switch to a small draft model (e.g., a tiny transformer) for the rest.

**Approach**: ngram-mod drafts 4-8 tokens → if accepted, a small model drafts the next 16-32 → verify all against the target model.

**Effort**: Very high — requires a draft model integration.

---

### 11. Beam Search for N-grams

Currently greedy: take the first prediction, chain it. If the first prediction is wrong, everything after is wrong.

**Approach**: At each step, try the top-K predictions (based on frequency). Explore multiple paths. Submit the longest accepted beam.

**Effort**: Medium — maintain K candidate sequences, evaluate all.

---

### 12. Prefix Trie (Instead of Hash Table)

A hash table gives one prediction per context. A trie gives **all** predictions for any prefix.

**Approach**: Build a trie of all n-grams. For context `[A, B, C, D]`, the trie tells you:
- `[A, B, C, D]` → `E` (freq 100)
- `[A, B, C]` → `F` (freq 50)
- `[A, B]` → `G` (freq 30)

Pick the longest context with highest frequency.

**Effort**: High — rewrite the data structure entirely.

---

### 14. Adaptive Eviction Threshold

The -20 threshold is fixed. During warm-up, entries haven't had many lookups yet. During steady state, they have.

**Approach**: Adjust the threshold based on table age:

```
First 100 accept cycles: threshold = -5 (evict quickly)
After 1000 cycles: threshold = -20 (stable)
During low acceptance: threshold = -10 (evict more aggressively)
```

**Effort**: Low — make threshold dynamic in `accept()`.

---

## Quick Wins (Low Effort, Good Impact)

| # | Idea | Effort | Expected Impact |
|---|------|--------|-----------------|
| 4 | Weighted increment by position | Low | **High** — rewards reliable early predictions |
| 2 | Confidence ratio eviction | Low | **Medium** — removes low-accuracy entries |
| 14 | Adaptive eviction threshold | Low | **Medium** — better warm-up behavior |
| 3 | Exponential decay | Low | **Medium** — recent predictions matter more |

---

### 15. Linear Probing for Hash Collisions

The current hash table has **no collision resolution** — two different n-grams hashing to the same slot means one silently overwrites the other. With 4M entries and real-world token distributions, collisions are inevitable.

**Approach**: When the target slot is occupied by a different n-gram, probe linearly (or with double hashing) to find the next free slot. This is a classic technique that dramatically reduces false misses.

```cpp
// In get():
size_t hash = compute_hash(context);
while (entries[hash] != EMPTY && !match(entries[hash], context)) {
    hash = (hash + 1) & mask;  // linear probe
}
```

**Effort**: Medium — modify `get()` and `add()` to probe, add context comparison. Requires storing the full n-gram key per entry (currently only stores the predicted token).

---

### 16. Adaptive n_min

The default `n_min = 48` is extremely aggressive — if the table misses at any of the first 48 speculative positions, the entire draft is discarded. During cold start or sparse tables, this means **zero** speculative tokens are produced.

**Approach**: Lower `n_min` when the table is cold or has low occupancy:

```
occupancy < 10%: n_min = 4
occupancy 10-30%: n_min = 8
occupancy > 30%: n_min = configured value (48)
```

Alternatively, track consecutive miss rate and adapt:

```
3+ consecutive misses at position < 10: temporarily lower n_min to 4
5+ consecutive hits: restore n_min
```

**Effort**: Low — add occupancy/miss-rate check before the n_min comparison in `draft_one()`.

---

### 17. Logit-Guided N-gram Validation

An n-gram prediction is just a guess. The target model's logits tell us how confident it is. If the n-gram predicts token X but the model gives it < 5% probability, the prediction is likely wrong — reject it early without wasting verification cycles.

**Approach**: After n-gram predicts a token, check the target model's top-K logits. If the predicted token is not in top-K or has probability below threshold, stop the chain:

```cpp
// After n-gram predicts token_id:
float prob = logits[token_id];
if (prob < 0.05) break;  // model doesn't agree, stop chain
```

**Effort**: Low — requires that the target model's logits are available during speculative decoding (they are in the current architecture).

---

### 18. Rare Token Bonus

Not all n-gram hits are equal. Predicting "the" correctly is easy (it's the most common token). Predicting a rare identifier correctly is much more valuable — the model is less likely to get it right on its own.

**Approach**: Weight the frequency increment inversely proportional to token frequency:

```cpp
// Global token frequency from corpus or running stats:
float bonus = 1.0f / (1.0f + log1p(token_global_freq[token_id]));
// Common token "the": bonus ≈ 0.01
// Rare token "initialize": bonus ≈ 0.5
// Very rare token "my_42_variable": bonus ≈ 1.0
weight = base_weight * (1.0f + bonus);
mod.inc(contexts[i], weight);
```

**Effort**: Medium — need global token frequency tracking (can be built during warm-up or from corpus stats).

---

### 19. Configurable Table Size

The 4M table size is hardcoded. For small models (7B) with short contexts, this is wasteful. For large models (70B+) with long contexts, it's insufficient and causes aggressive pruning.

**Approach**: Make table size a parameter, with sensible defaults based on model size:

```
model < 13B: 1M entries
model 13-34B: 4M entries
model > 34B: 8M entries
```

Or let the user specify it directly.

**Effort**: Low — parameterize the table size constant, add to `common_params_speculative_ngram_mod`.

---

### 20. LRU-Aware Eviction

Frequency tells you if an n-gram is good, but not if it's still relevant. An n-gram that was hot 10,000 tokens ago but hasn't been seen since the current section is noise.

**Approach**: Track last access time (or recency bit) per entry. During eviction, prefer evicting old entries over new ones at the same frequency level:

```cpp
// In prune_evict_negative:
// Among entries with freq < -20, evict the oldest first
// Among entries with freq > 0, keep the recently accessed ones
```

A simple approach: add a `last_access` field (uint32_t, global counter incremented each `accept()` call).

**Effort**: Medium — add `last_access` array, update in `get()`, use as tiebreaker in eviction.
