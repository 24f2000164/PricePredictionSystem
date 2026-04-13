<template>
  <div class="result" :class="{ 'result--cached': result.cached }">

    <!-- Cached badge -->
    <div v-if="result.cached" class="cached-badge">⚡ cached</div>

    <!-- ── Price hero ──────────────────────────────────────────── -->
    <div class="price-hero">
      <div>
        <p class="price-label">Predicted price</p>
        <p class="price-value">${{ result.predicted_price.toFixed(2) }}</p>
      </div>
      <div class="verdict-badge" :class="verdictClass">
        {{ result.value_verdict }}
      </div>
    </div>

    <!-- ── Explanation ─────────────────────────────────────────── -->
    <div class="section">
      <h3 class="section-title">Why this price</h3>
      <p class="explanation">{{ result.price_explanation }}</p>
    </div>

    <!-- ── Value reasoning ────────────────────────────────────── -->
    <div class="section">
      <h3 class="section-title">Verdict reasoning</h3>
      <p class="muted-text">{{ result.value_reasoning }}</p>
    </div>

    <!-- ── Target customers ───────────────────────────────────── -->
    <div class="section">
      <h3 class="section-title">Target customers</h3>
      <p class="muted-text">{{ result.target_customers }}</p>
    </div>

    <!-- ── Key features ───────────────────────────────────────── -->
    <div class="section">
      <h3 class="section-title">Key features</h3>
      <div class="tags">
        <span
          v-for="(feat, i) in result.key_features"
          :key="i"
          class="tag"
        >{{ feat }}</span>
      </div>
    </div>

    <!-- ── ML signals ─────────────────────────────────────────── -->
    <div class="section">
      <h3 class="section-title">ML model signals</h3>
      <div class="signals-grid">
        <div
          v-for="(val, key) in displaySignals"
          :key="key"
          class="signal-item"
        >
          <span class="signal-key">{{ formatKey(key) }}</span>
          <span class="signal-val" :class="signalClass(key, val)">{{ formatVal(key, val) }}</span>
        </div>
      </div>
    </div>

  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  result: { type: Object, required: true },
})

// Only show the most meaningful signals in the UI
const displaySignals = computed(() => {
  const s = props.result.top_signals
  return {
    category:         s.category,
    pack_count:       s.pack_count,
    premium_score:    s.premium_score,
    bulk_score:       s.bulk_score,
    brand_score:      s.brand_score,
    has_special_diet: s.has_special_diet,
    quality_indicator:s.quality_indicator,
    price_tier:       s.price_tier,
  }
})

const verdictClass = computed(() => ({
  'verdict--good':    props.result.value_verdict === 'Good value',
  'verdict--fair':    props.result.value_verdict === 'Fair price',
  'verdict--premium': props.result.value_verdict === 'Premium priced',
}))

function formatKey(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function formatVal(key, val) {
  if (key === 'has_special_diet') return val ? 'Yes' : 'No'
  if (key === 'price_tier') {
    if (val > 1)  return 'Premium'
    if (val < -1) return 'Budget'
    return 'Mid-range'
  }
  if (typeof val === 'number') return Number.isInteger(val) ? val : val.toFixed(1)
  return val
}

function signalClass(key, val) {
  if (key === 'premium_score' && val > 2) return 'val--high'
  if (key === 'bulk_score'    && val > 1) return 'val--accent'
  if (key === 'has_special_diet' && val)  return 'val--high'
  if (key === 'price_tier' && val > 1)    return 'val--high'
  if (key === 'price_tier' && val < -1)   return 'val--low'
  return ''
}
</script>

<style scoped>
.result {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  animation: fadeUp 0.35s ease both;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Cached badge ────────────────────────────────────────────── */
.cached-badge {
  position: absolute;
  top: -0.6rem;
  right: 0;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 20px;
  padding: 2px 10px;
  color: var(--muted);
}

/* ── Price hero ──────────────────────────────────────────────── */
.price-hero {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.5rem;
  gap: 1rem;
}
.price-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin-bottom: 0.25rem;
}
.price-value {
  font-family: var(--serif);
  font-size: 3rem;
  color: var(--accent);
  line-height: 1;
}

/* ── Verdict badge ───────────────────────────────────────────── */
.verdict-badge {
  font-size: 0.78rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  padding: 6px 14px;
  border-radius: 20px;
  border: 1px solid;
  white-space: nowrap;
}
.verdict--good    { color: #7ee87e; border-color: #7ee87e40; background: #7ee87e12; }
.verdict--fair    { color: #f0c84a; border-color: #f0c84a40; background: #f0c84a12; }
.verdict--premium { color: #c084f0; border-color: #c084f040; background: #c084f012; }

/* ── Section ─────────────────────────────────────────────────── */
.section {
  border-top: 1px solid var(--border);
  padding-top: 1.25rem;
}
.section-title {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin-bottom: 0.6rem;
  font-weight: 500;
}
.explanation {
  font-size: 0.95rem;
  color: var(--text);
  line-height: 1.7;
}
.muted-text {
  font-size: 0.9rem;
  color: #aaa;
  line-height: 1.6;
}

/* ── Tags ────────────────────────────────────────────────────── */
.tags { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.tag {
  font-size: 0.8rem;
  padding: 4px 12px;
  border-radius: 20px;
  background: var(--surface2);
  border: 1px solid var(--border2);
  color: var(--text);
}

/* ── Signals grid ────────────────────────────────────────────── */
.signals-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem 1rem;
}
.signal-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.83rem;
  padding: 6px 10px;
  background: var(--surface2);
  border-radius: 6px;
}
.signal-key { color: var(--muted); }
.signal-val { font-weight: 500; color: var(--text); }
.val--high   { color: var(--accent); }
.val--low    { color: #f0c84a; }
.val--accent { color: var(--info); }
</style>