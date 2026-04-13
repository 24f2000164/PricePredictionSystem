<template>
  <div class="app">

    <!-- ── Header ──────────────────────────────────────────────── -->
    <header class="header">
      <div class="header-inner">
        <div class="logo">
          <span class="logo-mark">IQ</span>
          <span class="logo-text">Price<em>IQ</em></span>
        </div>
        <p class="tagline">Multimodal price intelligence · CLIP + Gemini</p>
      </div>
    </header>

    <!-- ── Main ────────────────────────────────────────────────── -->
    <main class="main">
      <div class="layout">

        <!-- LEFT PANEL — input form -->
        <section class="panel panel-input">
          <h2 class="panel-title">Analyse product</h2>

          <div class="field">
            <label class="label">Catalog content</label>
            <textarea
              v-model="form.catalog_content"
              class="textarea"
              placeholder="Paste the full product title + description + pack quantity here…"
              rows="10"
            />
          </div>

          <div class="field">
            <label class="label">Image URL</label>
            <input
              v-model="form.image_link"
              class="input"
              type="url"
              placeholder="https://example.com/product-image.jpg"
            />
          </div>

          <!-- Image preview -->
          <div v-if="form.image_link" class="preview-wrap">
            <img
              :src="form.image_link"
              class="preview-img"
              alt="Product preview"
              @error="imgError = true"
              v-show="!imgError"
            />
            <p v-if="imgError" class="preview-error">Could not load image preview</p>
          </div>

          <button
            class="btn-primary"
            :disabled="loading || !form.catalog_content || !form.image_link"
            @click="analyse"
          >
            <span v-if="!loading">Analyse &amp; price →</span>
            <span v-else class="loading-row">
              <span class="spinner" />
              Running inference…
            </span>
          </button>

          <p v-if="error" class="error-msg">{{ error }}</p>
        </section>

        <!-- RIGHT PANEL — results -->
        <section class="panel panel-result">
          <!-- Empty state -->
          <div v-if="!result && !loading" class="empty-state">
            <div class="empty-icon">₹</div>
            <p>Submit a product to see the ML prediction and Gemini explanation.</p>
          </div>

          <!-- Loading skeleton -->
          <div v-if="loading" class="skeleton-wrap">
            <div class="skeleton sk-price" />
            <div class="skeleton sk-line" />
            <div class="skeleton sk-line sk-short" />
            <div class="skeleton sk-line" />
            <div class="skeleton sk-line sk-short" />
          </div>

          <!-- Result card -->
          <ResultCard v-if="result && !loading" :result="result" />
        </section>

      </div>
    </main>

    <!-- ── Footer ──────────────────────────────────────────────── -->
    <footer class="footer">
      <p>CLIP ViT-L/14 · LoRA fine-tuned · 40 tabular features · Gemini Flash explanations</p>
    </footer>

  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import ResultCard from './components/ResultCard.vue'

const form = ref({ catalog_content: '', image_link: '' })
const result  = ref(null)
const loading = ref(false)
const error   = ref('')
const imgError = ref(false)

watch(() => form.value.image_link, () => { imgError.value = false })

async function analyse() {
  if (!form.value.catalog_content.trim() || !form.value.image_link.trim()) return
  loading.value = true
  error.value   = ''
  result.value  = null

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        catalog_content: form.value.catalog_content,
        image_link:      form.value.image_link,
      }),
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || 'Server error')
    }

    result.value = await res.json()
  } catch (e) {
    error.value = `Error: ${e.message}`
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
/* ── Layout ──────────────────────────────────────────────────── */
.app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* ── Header ──────────────────────────────────────────────────── */
.header {
  border-bottom: 1px solid var(--border);
  padding: 0 2rem;
}
.header-inner {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  gap: 1.5rem;
  height: 64px;
}
.logo {
  display: flex;
  align-items: center;
  gap: 10px;
}
.logo-mark {
  background: var(--accent);
  color: #000;
  font-family: var(--serif);
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  padding: 4px 8px;
  border-radius: 6px;
}
.logo-text {
  font-family: var(--serif);
  font-size: 1.4rem;
  color: var(--text);
}
.logo-text em {
  font-style: italic;
  color: var(--accent);
}
.tagline {
  font-size: 0.78rem;
  color: var(--muted);
  margin-left: auto;
  letter-spacing: 0.03em;
}

/* ── Main ────────────────────────────────────────────────────── */
.main {
  flex: 1;
  padding: 2.5rem 2rem;
}
.layout {
  max-width: 1200px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 440px 1fr;
  gap: 2rem;
  align-items: start;
}

/* ── Panels ──────────────────────────────────────────────────── */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 2rem;
}
.panel-title {
  font-family: var(--serif);
  font-size: 1.4rem;
  font-weight: 400;
  margin-bottom: 1.5rem;
  color: var(--text);
}

/* ── Form elements ───────────────────────────────────────────── */
.field { margin-bottom: 1.2rem; }

.label {
  display: block;
  font-size: 0.78rem;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.5rem;
}

.textarea, .input {
  width: 100%;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  padding: 0.75rem 1rem;
  resize: vertical;
  transition: border-color 0.2s;
  line-height: 1.6;
}
.textarea:focus, .input:focus {
  outline: none;
  border-color: var(--accent);
}
.textarea::placeholder, .input::placeholder { color: #555; }

/* ── Image preview ───────────────────────────────────────────── */
.preview-wrap {
  margin-bottom: 1.2rem;
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border);
  background: var(--surface2);
  max-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.preview-img {
  max-height: 200px;
  max-width: 100%;
  object-fit: contain;
}
.preview-error {
  color: var(--muted);
  font-size: 0.85rem;
  padding: 1rem;
}

/* ── Primary button ──────────────────────────────────────────── */
.btn-primary {
  width: 100%;
  background: var(--accent);
  color: #000;
  border: none;
  border-radius: var(--radius);
  padding: 0.85rem 1.5rem;
  font-weight: 500;
  font-size: 0.95rem;
  cursor: pointer;
  transition: background 0.2s, transform 0.1s;
  letter-spacing: 0.02em;
}
.btn-primary:hover:not(:disabled) { background: var(--accent2); }
.btn-primary:active:not(:disabled) { transform: scale(0.99); }
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

.loading-row { display: flex; align-items: center; justify-content: center; gap: 0.6rem; }
.spinner {
  width: 16px; height: 16px;
  border: 2px solid #00000040;
  border-top-color: #000;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  display: inline-block;
}
@keyframes spin { to { transform: rotate(360deg); } }

.error-msg {
  margin-top: 0.75rem;
  font-size: 0.85rem;
  color: var(--danger);
}

/* ── Empty state ─────────────────────────────────────────────── */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  text-align: center;
  gap: 1rem;
}
.empty-icon {
  font-family: var(--serif);
  font-size: 3.5rem;
  color: var(--border2);
  line-height: 1;
}
.empty-state p { color: var(--muted); font-size: 0.9rem; max-width: 260px; }

/* ── Skeleton loader ─────────────────────────────────────────── */
.skeleton-wrap { display: flex; flex-direction: column; gap: 0.8rem; padding: 0.5rem; }
.skeleton {
  background: linear-gradient(90deg, var(--surface2) 25%, var(--border) 50%, var(--surface2) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
  border-radius: var(--radius);
}
.sk-price  { height: 72px; border-radius: var(--radius-lg); }
.sk-line   { height: 18px; }
.sk-short  { width: 60%; }
@keyframes shimmer { to { background-position: -200% 0; } }

/* ── Footer ──────────────────────────────────────────────────── */
.footer {
  border-top: 1px solid var(--border);
  padding: 1rem 2rem;
  text-align: center;
  font-size: 0.75rem;
  color: var(--muted);
  letter-spacing: 0.04em;
}

/* ── Responsive ──────────────────────────────────────────────── */
@media (max-width: 860px) {
  .layout { grid-template-columns: 1fr; }
  .tagline { display: none; }
}
</style>