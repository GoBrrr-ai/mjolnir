/* Mjolnir benchmarks — the GitHub-Pages app (plain JS: no framework, no build).
 *
 * Boot fetches the small manifest.json, renders the filters, then lazily
 * fetches ONE payload per measured (platform, model) pair from
 * data/<platform>/<model-short>.json. The URL hash is the whole state —
 * #platform=<p>&model=<m>&config=<q>&stage=<s>&compare=1 — so every view is a
 * shareable permalink. With no hash the hero shows the manifest's default pair
 * (thor, Qwen/Qwen3.8-27B). Plotly comes from the CDN script tag in
 * index.html; colors come from app.css custom properties, so the theme
 * switch restyles every chart. Compare mode fetches a second payload for the
 * same (model, config) on another platform — there is no server side.
 */
"use strict";

const HASH_KEYS = ["platform", "model", "config", "stage", "compare"];
const DEFAULT_PAIR = {platform: "thor", model: "Qwen/Qwen3.8-27B"};

const NAMES = {"mjolnir": "Mjolnir · e2m3 KV", "mjolnir-fp8": "Mjolnir · fp8 KV",
               "fi": "FlashInfer · fp8 KV", "fi-hnd": "FlashInfer · HND", "fa2": "FlashAttention 2 · bf16 KV"};
const STYLE = {"mjolnir": ["accent", "circle", true], "mjolnir-fp8": ["accent", "square-open", false],
               "fi": ["fi", "diamond", true], "fi-hnd": ["fi", "diamond-open", false], "fa2": ["fa2", "triangle-up", true]};
const E2E_METRICS = {tg: "decode tokens/s", pp: "prefill tokens/s", ttft: "TTFT (ms)", itl: "ITL (ms)"};

let MANIFEST = null;                       // manifest.json (the small index)
let PAYLOAD = null;                        // the selected pair's payload
let LOADED = null;                         // the entry PAYLOAD was fetched for
const CACHE = new Map();                   // payload url -> parsed payload (compare mode)
const REDRAW = [];                         // chart rebuild hooks

// ── state (the URL hash) ────────────────────────────────────────────────────

function parseHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  return {platform: q.get("platform") || "", model: q.get("model") || "",
          config: q.get("config") || "", stage: q.get("stage") || "",
          compare: q.get("compare") === "1"};
}
let state = parseHash();

function writeHash() {
  const q = new URLSearchParams();
  for (const k of HASH_KEYS) {
    const v = k === "compare" ? (state.compare ? "1" : "") : state[k];
    if (v) q.set(k, v);
  }
  const h = q.toString();
  history.replaceState(null, "", h ? "#" + h : location.pathname + location.search);
  state = parseHash();
}

function setState(patch) {
  Object.assign(state, patch);
  writeHash();
  refresh();
}

// ── the palette (app.css custom properties — theme.py's validated pairs) ────

function C() {
  const s = getComputedStyle(document.documentElement);
  const v = n => s.getPropertyValue(n).trim();
  return {bg: v("--bg"), panel: v("--panel"), text: v("--text"), dim: v("--dim"), grid: v("--grid"),
          mjolnir: v("--accent"), fi: v("--fi"), fa2: v("--fa2"), neutral: v("--neutral")};
}

// ── small helpers ───────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
// Escape a string for safe interpolation into innerHTML — mandatory for
// anything derived from location.hash (attacker-controlled via a permalink).
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}
const LS_LABEL = L => (L >= 1024 && L % 1024 === 0 ? L / 1024 + "K" : String(L));
function fmtUs(t) { return t == null ? "—" : t >= 1000 ? (t / 1000).toFixed(2) + " ms" : t.toFixed(1) + " µs"; }
function geomean(xs) {
  xs = xs.filter(x => x > 0);
  return xs.length ? Math.exp(xs.reduce((a, x) => a + Math.log(x), 0) / xs.length) : null;
}
function byId(doc) { const o = {}; (doc.series || []).forEach(s => (o[s.id] = s)); return o; }
function pm(s, f) { const o = {}; (s.points || []).forEach(p => { if (p[f || "total_us"] != null) o[p.L] = p[f || "total_us"]; }); return o; }
function pairEntry(platform, model) {
  return (MANIFEST.pairs || []).find(p => p.platform === platform && p.model === model);
}
function payloadUrl(entry) {
  return "data/" + encodeURIComponent(entry.platform) + "/" + encodeURIComponent(entry.short) + ".json";
}
async function fetchPayload(entry) {
  const url = payloadUrl(entry);
  if (!CACHE.has(url)) CACHE.set(url, await (await fetch(url)).json());
  return CACHE.get(url);
}
function familyColor(c, cfg) {
  return /MJOLNIR|mjolnir/.test(cfg) ? c.mjolnir : /FA2/.test(cfg) ? c.fa2 : c.fi;
}

// ── plotly scaffolding ──────────────────────────────────────────────────────

function layout(extra) {
  const c = C();
  return Object.assign({
    paper_bgcolor: c.panel, plot_bgcolor: c.panel,
    font: {color: c.text, family: "system-ui, sans-serif", size: 13},
    margin: {l: 64, r: 24, t: 24, b: 56}, hovermode: "x unified",
    hoverlabel: {bgcolor: c.bg, bordercolor: c.grid, font: {color: c.text}},
    legend: {orientation: "h", y: -0.2, font: {color: c.text}},
    xaxis: {gridcolor: c.grid, zeroline: false, color: c.dim},
    yaxis: {gridcolor: c.grid, zeroline: false, color: c.dim},
  }, extra || {});
}
function lAxis(Ls) {
  return {type: "log", tickvals: Ls, ticktext: Ls.map(LS_LABEL), title: "context length L (tokens)",
          gridcolor: C().grid, color: C().dim};
}
function line(sid, xs, ys, name, hover) {
  const c = C(), st = STYLE[sid] || ["accent", "circle", true];
  return {x: xs, y: ys, name: name || NAMES[sid] || sid, mode: "lines+markers", type: "scatter",
          line: {color: c[st[0]], width: 2},
          marker: {symbol: st[1], size: 9, color: st[2] ? c[st[0]] : c.panel,
                   line: {color: c[st[0]], width: 2}},
          hovertemplate: hover || "%{y:.2f}<extra>%{fullData.name}</extra>"};
}
function noiseShapes(c) {
  const n = (MANIFEST ? MANIFEST.noise_pct : 3) / 100;
  return [{type: "rect", xref: "paper", x0: 0, x1: 1, y0: 1 - n, y1: 1 + n, fillcolor: c.grid,
           line: {width: 0}, layer: "below"},
          {type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: {color: c.fi, width: 1.5}}];
}
function draw(id, traces, opts) {
  Plotly.react(id, traces, layout(opts), {responsive: true, displaylogo: false});
}
function table(header, rows) {
  return '<div class="tablewrap"><table><thead><tr>' + header.map(h => "<th>" + h + "</th>").join("") +
    "</tr></thead><tbody>" + rows.map(r => "<tr>" + r.map(x => "<td>" + x + "</td>").join("") + "</tr>").join("") +
    "</tbody></table></div>";
}
function section(parent, id, title, what, build, tableHtml) {
  const s = el("section");
  s.id = id;
  s.appendChild(el("h2", null, title));
  s.appendChild(el("p", "what", what));
  const chart = el("div", "chart");
  chart.id = "c-" + id;
  s.appendChild(chart);
  if (tableHtml) s.appendChild(el("details", null, "<summary>data table</summary>" + tableHtml));
  parent.appendChild(s);
  build(chart.id);
  REDRAW.push(() => build(chart.id));
}
function oursIds(doc) {
  return (doc.series || []).map(s => s.id)
    .filter(id => ($("#filter-family").value !== "mjolnir" || id.startsWith("mjolnir")));
}

// ── stage views (one function per stage shape) ──────────────────────────────

function sweepView(parent, doc) {
  const by = byId(doc);
  if (!by.fi) return;
  const fi = pm(by.fi), Ls = Object.keys(fi).map(Number).sort((a, b) => a - b);
  const speedup = sid => {
    const o = pm(by[sid]);
    return line(sid, Ls.filter(L => o[L]), Ls.filter(L => o[L]).map(L => fi[L] / o[L]), null,
                "%{y:.2f}×<extra>%{fullData.name}</extra>");
  };
  section(parent, "sweep", "Mjolnir vs FlashInfer vs FlashAttention 2",
    "Speedup over FlashInfer at every context length (ncu kernel time; gray band = ±" + MANIFEST.noise_pct +
    "% noise).", id => draw(id, oursIds(doc).filter(s => s !== "fi" && by[s]).map(speedup),
      {xaxis: lAxis(Ls), yaxis: {title: "speedup over FlashInfer (×)", gridcolor: C().grid, color: C().dim,
                                 rangemode: "tozero"}, shapes: noiseShapes(C())}),
    table(["L"].concat((doc.series || []).map(s => s.label)),
          Ls.map(L => [LS_LABEL(L)].concat((doc.series || []).map(s => fmtUs(pm(s)[L]))))));
  section(parent, "latency", "Decode latency per step",
    "Time for one decode step over the whole KV cache (log scale, lower is faster).",
    id => draw(id, oursIds(doc).filter(s => by[s]).map(sid => {
      const o = pm(by[sid]), xs = Object.keys(o).map(Number).sort((a, b) => a - b);
      return line(sid, xs, xs.map(L => o[L]), null, "%{y:.1f} µs<extra>%{fullData.name}</extra>");
    }), {xaxis: lAxis(Ls), yaxis: {type: "log", title: "µs per step", gridcolor: C().grid, color: C().dim}}));
  section(parent, "bandwidth", "Effective KV bandwidth",
    "Each leg's real KV bytes ÷ decode time, against the 273 GB/s LPDDR5X peak.",
    id => draw(id, oursIds(doc).filter(s => by[s]).map(sid => {
      const o = pm(by[sid], "gbps"), xs = Object.keys(o).map(Number).sort((a, b) => a - b);
      return line(sid, xs, xs.map(L => o[L]), null, "%{y:.0f} GB/s<extra>%{fullData.name}</extra>");
    }), {xaxis: lAxis(Ls), yaxis: {title: "GB/s", range: [0, 300], gridcolor: C().grid, color: C().dim},
         shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: 273, y1: 273, line: {color: C().dim, width: 1.2}}]}));
}

function ratioView(parent, doc, title, what) {
  const by = byId(doc);
  if (!by.fi) return;
  const fi = pm(by.fi), Ls = Object.keys(fi).map(Number).sort((a, b) => a - b);
  section(parent, doc.stage, title, what, id => draw(id,
    oursIds(doc).filter(s => s !== "fi" && by[s]).map(sid => {
      const o = pm(by[sid]);
      return line(sid, Ls.filter(L => o[L]), Ls.filter(L => o[L]).map(L => fi[L] / o[L]), by[sid].label,
                  "%{y:.2f}×<extra>%{fullData.name}</extra>");
    }),
    {xaxis: lAxis(Ls), yaxis: {title: "speedup over FlashInfer NHD (×)", gridcolor: C().grid, color: C().dim},
     shapes: noiseShapes(C())}));
}

function heatView(parent, doc, title, what) {
  const by = byId(doc);
  if (!by.fi) return;
  const fi = pm(by.fi), Ls = Object.keys(fi).map(Number).sort((a, b) => a - b);
  const vs = (doc.series || []).filter(s => s.kind === "t5");
  if (!vs.length) return;
  const z = vs.map(s => { const o = pm(s); return Ls.map(L => (o[L] ? fi[L] / o[L] : null)); });
  const zlog = z.map(r => r.map(v => (v ? Math.max(-1, Math.min(1, Math.log2(v))) : null)));
  section(parent, doc.stage, title, what, id => {
    const c = C();
    draw(id, [{type: "heatmap", x: Ls.map(LS_LABEL), y: vs.map(s => s.label), z: zlog, customdata: z,
               zmin: -1, zmax: 1, colorscale: [[0, c.fi], [0.5, c.neutral], [1, c.mjolnir]], xgap: 2, ygap: 2,
               showscale: false, text: z.map(r => r.map(v => (v ? v.toFixed(2) + "×" : ""))),
               texttemplate: "%{text}", textfont: {color: c.text},
               hovertemplate: "%{y} @ L=%{x}: %{customdata:.3f}× vs FlashInfer<extra></extra>"}],
         {margin: {l: 150, r: 24, t: 16, b: 48}, height: 120 + 44 * vs.length,
          xaxis: {color: c.dim, title: "context length L"}, yaxis: {color: c.text, autorange: "reversed"},
          hovermode: "closest"});
  }, table(["variant", "flags"].concat(Ls.map(LS_LABEL)),
           vs.map((s, i) => [s.label, "<code>" + (s.flags || "(default)") + "</code>"]
                           .concat(z[i].map(v => (v ? v.toFixed(2) + "×" : "—"))))));
}

function quantView(parent, doc) {
  const by = {}, bits = {};
  (doc.rows || []).forEach(r => {
    if (r.rel_l2 != null) { (by[r.format] = by[r.format] || []).push(r); bits[r.format] = r.bits; }
  });
  const fmts = Object.keys(by).sort((a, b) => (bits[b] || 0) - (bits[a] || 0));
  if (!fmts.length) return;
  const mean = f => by[f].reduce((a, r) => a + r.rel_l2, 0) / by[f].length;
  section(parent, "quant", "KV format accuracy",
    "Relative L2 error of the kernel output vs attention over unquantized KV — bar = mean over L, whisker = min–max.",
    id => {
      const c = C(), m = fmts.map(mean);
      draw(id, [{type: "bar", x: fmts.map(f => f + " · " + bits[f] + " bit"), y: m,
                 marker: {color: c.mjolnir},
                 error_y: {type: "data", symmetric: false,
                           array: fmts.map((f, i) => Math.max(...by[f].map(r => r.rel_l2)) - m[i]),
                           arrayminus: fmts.map((f, i) => m[i] - Math.min(...by[f].map(r => r.rel_l2))),
                           color: c.dim, thickness: 1.5, width: 6},
                 customdata: fmts.map(f => by[f].map(r => LS_LABEL(r.L) + ": " + r.rel_l2.toFixed(4)).join("<br>")),
                 hovertemplate: "%{x}<br>mean %{y:.4f}<br>%{customdata}<extra></extra>"}],
           {yaxis: {title: "relative L2 error (lower is better)", gridcolor: c.grid, color: c.dim},
            xaxis: {color: c.text}, hovermode: "closest", showlegend: false});
    },
    table(["format", "bits"].concat(by[fmts[0]].map(r => LS_LABEL(r.L))),
          fmts.map(f => [f, bits[f]].concat(by[f].map(r => r.rel_l2.toFixed(4))))));
}

function pdlView(parent, doc) {
  const rows = (doc.rows || []).filter(r => r.delta_pct != null);
  const bs = [...new Set(rows.map(r => r.batches))];
  section(parent, "pdl", "Programmatic dependent launch",
    "Wall-clock change with PDL on — orange = faster, blue = slower; hatched = batch 8; gray band = ±" +
    MANIFEST.noise_pct + "% noise.", id => {
      const c = C();
      draw(id, bs.map((b, i) => {
        const rs = rows.filter(r => r.batches === b);
        return {type: "bar", name: "batch " + b, x: rs.map(r => LS_LABEL(r.L)), y: rs.map(r => r.delta_pct),
                marker: {color: rs.map(r => (r.delta_pct < 0 ? c.mjolnir : c.fi)),
                         pattern: {shape: i ? "/" : "", fgcolor: c.panel, solidity: 0.35}},
                hovertemplate: "batch " + b + " @ L=%{x}: %{y:+.1f}%<extra></extra>"};
      }),
           {barmode: "group", yaxis: {title: "Δ wall clock (%)", gridcolor: c.grid, color: c.dim},
            xaxis: {color: c.dim}, hovermode: "closest",
            shapes: [{type: "rect", xref: "paper", x0: 0, x1: 1, y0: -7, y1: 7, fillcolor: c.grid,
                      line: {width: 0}, layer: "below"}]});
    });
}

function profView(parent, doc) {
  const ps = (doc.profiles || []).filter(p => p.ok);
  if (!ps.length) return;
  const labs = [...new Set(ps.map(p => p.label))], base = labs[0];
  const Ls = [...new Set(ps.map(p => p.L))].sort((a, b) => a - b);
  const get = (lab, L) => ps.find(p => p.label === lab && p.L === L);
  section(parent, "fp6prof", "Bytes vs time per KV format",
    "Full ncu profiles: L2 bytes read and kernel time relative to " + base + " (the ideal is the bits ratio).",
    id => {
      const c = C(), traces = [];
      labs.slice(1).forEach((lab, i) => {
        const t = Ls.map(L => { const b = get(base, L), p = get(lab, L);
                                return b && p ? p.metrics["gpu__time_duration.sum"] / b.metrics["gpu__time_duration.sum"] : null; });
        const s = Ls.map(L => { const b = get(base, L), p = get(lab, L);
                                const k = "lts__t_sectors_op_read.sum"; return b && p && b.metrics[k] ? p.metrics[k] / b.metrics[k] : null; });
        traces.push({type: "bar", name: lab + " · kernel time", x: Ls.map(LS_LABEL), y: t,
                     marker: {color: c.mjolnir, opacity: i ? 0.55 : 1}});
        traces.push({type: "bar", name: lab + " · L2 bytes", x: Ls.map(LS_LABEL), y: s,
                     marker: {color: c.dim, opacity: i ? 0.55 : 1}});
      });
      draw(id, traces, {barmode: "group", yaxis: {title: "ratio vs " + base, range: [0, 1.15],
        gridcolor: c.grid, color: c.dim}, xaxis: {color: c.dim}, hovermode: "x unified"});
    });
}

function e2eBars(mount, rows, id, title, what) {
  const metric = $("#filter-metric").value || "tg";
  const keys = Object.keys(rows).filter(cfg => {
    if (state.config && cfg !== state.config) return false;
    return $("#filter-family").value !== "mjolnir" || /MJOLNIR|mjolnir/.test(cfg);
  });
  if (!keys.length) return;
  const cellLabel = x => "c" + x.concurrency + " · " + (x.context ? LS_LABEL(x.context) + " ctx" : "no ctx");
  section(mount, id, title, what, cid => {
    const c = C();
    draw(cid, keys.map((k, i) => {
      const cs = rows[k].cells.filter(x => x[metric] != null);
      return {type: "bar", name: k, x: cs.map(cellLabel), y: cs.map(x => x[metric]),
              marker: {color: familyColor(c, k), opacity: i && keys.length > 1 ? 0.55 : 1},
              hovertemplate: "%{y:.1f}<extra>%{fullData.name}</extra>"};
    }),
         {barmode: "group", yaxis: {title: E2E_METRICS[metric] || metric, gridcolor: c.grid, color: c.dim},
          xaxis: {color: c.dim}});
  }, table(["config", "cell", E2E_METRICS[metric] || metric],
           keys.flatMap(k => rows[k].cells.map(x => [k, cellLabel(x), x[metric]]))));
}

function e2eView(parent, payload, id, title, what) {
  if (!Object.keys(payload.e2e).length) return;
  e2eBars(parent, payload.e2e, id || "e2e", title || "End-to-end serving",
          what || "The official vLLM image (native backends) vs the Mjolnir image, same model and config " +
                  "(llama-benchy, gated clean windows).");
}

function genericView(parent, doc) {
  const rows = doc.rows || [];
  if (!rows.length) return;
  const cols = [...new Set(rows.flatMap(r => Object.keys(r)))].filter(k => typeof rows[0][k] !== "object");
  const meta = MANIFEST.stages[doc.stage] || {};
  section(parent, doc.stage, meta.title || doc.stage, meta.summary || "", () => {},
          table(cols, rows.map(r => cols.map(c => (r[c] == null ? "—" : String(r[c]))))));
}

function specView(parent, doc) {
  const all = doc.rows || [];
  const rows = state.config ? all.filter(r => r.config === state.config) : all;
  if (!rows.length) return genericView(parent, doc);
  const by = {};
  rows.forEach(r => { (by[r.config] = by[r.config] || []).push(r); });
  section(parent, "spec", "Speculative decoding",
    "Decode tokens/s and per-draft acceptance for every config (K = drafts per verify step).", id => {
      const c = C();
      draw(id, Object.keys(by).map(cfg => {
        const rs = by[cfg];
        return {type: "bar", name: cfg, x: rs.map(r => "K=" + r.k + " · c" + r.concurrency),
                y: rs.map(r => r.decode_tps), marker: {color: familyColor(c, cfg)},
                hovertemplate: "%{y:.1f} tok/s<extra>%{fullData.name}</extra>"};
      }),
           {barmode: "group", yaxis: {title: "decode tokens/s", gridcolor: c.grid, color: c.dim},
            xaxis: {color: c.dim}});
    },
    table(["config", "conc", "K", "decode t/s", "acceptance"],
          rows.map(r => [r.config, r.concurrency, r.k, r.decode_tps, r.acceptance_length])));
}

function kvcacheView(parent, doc) {
  const rows = doc.rows || [];
  if (!rows.length) return;
  section(parent, "kv-cache-size", "KV cache capacity",
    "Tokens of KV each config fits (vLLM's startup sizing) and the concurrency that buys per context.", id => {
      const c = C();
      draw(id, [{type: "bar", x: rows.map(r => r.id), y: rows.map(r => r.kv_tokens),
                 marker: {color: rows.map(r => (r.side === "mjolnir" ? c.mjolnir : c.fi))},
                 hovertemplate: "%{x}: %{y:,.0f} tokens<extra></extra>"}],
           {yaxis: {title: "KV tokens", gridcolor: c.grid, color: c.dim}, xaxis: {color: c.dim}});
    },
    table(["id", "side", "kv dtype", "KV tokens", "max conc", "max len"],
          rows.map(r => [r.id, r.side, r.kv_dtype, r.kv_tokens, r.max_concurrency, r.max_model_len])));
}

function correctnessView(parent, doc) {
  const cs = doc.cases || [];
  if (!cs.length) return;
  const ok = cs.filter(c => c.passed).length;
  section(parent, "correctness", "Correctness matrix",
    ok + " / " + cs.length + " checks pass against the bit-faithful emulator.", () => {},
    table(["case", "q", "passed"], cs.map(c => [c.label, c.q, c.passed ? "✅" : "❌"])));
}

const VIEWS = {
  sweep: (parent, doc) => sweepView(parent, doc),
  hndsweep: (parent, doc) => ratioView(parent, doc, "Head-major KV pages",
    "Same data on NHD vs HND pages — FlashInfer gains a little, Mjolnir more."),
  kvfmt: (parent, doc) => heatView(parent, doc, "KV layouts × formats",
    "Speedup over FlashInfer per KV format and page layout."),
  quant: (parent, doc) => quantView(parent, doc),
  ring: (parent, doc) => heatView(parent, doc, "KV ring depth", "TMA → MMA ring depth (--kv_stages)."),
  prefetch: (parent, doc) => heatView(parent, doc, "L2 prefetch", "TMA L2 prefetch distance (--l2_prefetch)."),
  pdl: (parent, doc) => pdlView(parent, doc),
  ab: (parent, doc) => heatView(parent, doc, "Flag A/B", "Harness flag sets (default: evict-first × PDL)."),
  fp6prof: (parent, doc) => profView(parent, doc),
  spec: (parent, doc) => specView(parent, doc),
  "kv-cache-size": (parent, doc) => kvcacheView(parent, doc),
  correctness: (parent, doc) => correctnessView(parent, doc),
};

// ── host / run-age narrow the stage docs; the metric select picks the e2e y ──

function passFilters(doc) {
  const host = $("#filter-host").value, age = Number($("#filter-age").value);
  if (host && doc.host !== host) return false;
  if (age) {
    const t = Date.parse(doc.ts || "");
    if (!Number.isNaN(t) && Date.now() - t > age * 86400000) return false;
  }
  return true;
}

function metricOptions(stage) {
  const sel = $("#filter-metric");
  const opts = stage === "e2e" ? Object.keys(E2E_METRICS) : [];
  const cur = sel.value;
  sel.innerHTML = "";
  opts.forEach(m => {
    const o = el("option", null, E2E_METRICS[m]);
    o.value = m;
    sel.appendChild(o);
  });
  if (opts.includes(cur)) sel.value = cur;
  sel.closest(".fgroup").style.display = opts.length ? "" : "none";
}

// ── rendering ───────────────────────────────────────────────────────────────

function render() {
  REDRAW.length = 0;
  const views = $("views");
  views.innerHTML = "";
  if (!PAYLOAD) return;
  if (!state.stage) return hero(views);
  if (state.stage === "e2e") return e2eView(views, PAYLOAD);
  const doc = PAYLOAD.stages[state.stage];
  if (!doc || !passFilters(doc)) {
    return views.appendChild(el("p", "empty", "No " + esc(state.stage) + " run for this filter selection yet."));
  }
  (VIEWS[state.stage] || genericView)(views, doc);
}

function hero(views) {
  const doc = PAYLOAD.stages.sweep, out = [];
  if (doc) {
    const by = byId(doc), ours = by.mjolnir || by["mjolnir-fp8"];
    if (ours && by.fi) {
      const fi = pm(by.fi), o = pm(ours), sp = Object.keys(o).filter(L => fi[L]).map(L => fi[L] / o[L]);
      out.push([geomean(sp).toFixed(2) + "×", "geomean speedup over FlashInfer (all L)"]);
      out.push([Math.max(...sp).toFixed(2) + "×", "best speedup over FlashInfer"]);
      if (by.fa2) {
        const fa = pm(by.fa2), s2 = Object.keys(o).filter(L => fa[L]).map(L => fa[L] / o[L]);
        out.push([geomean(s2).toFixed(2) + "×", "geomean speedup over FlashAttention 2"]);
      }
    }
  }
  const corr = PAYLOAD.stages.correctness;
  if (corr) {
    const cs = corr.cases || [];
    out.push([cs.filter(c => c.passed).length + " / " + cs.length, "correctness checks passing"]);
  }
  $("tiles").innerHTML = out.length
    ? out.map(t => "<div class='tile'><b>" + t[0] + "</b><span>" + t[1] + "</span></div>").join("")
    : "<div class='tile'><b>—</b><span>no verify runs yet — run mjolnir verify bench</span></div>";
  if (doc) sweepView(views, doc);
  e2eView(views, PAYLOAD);
}

// ── compare: same (model, config) across platforms, no server side ──────────

async function compareView(views) {
  const others = (MANIFEST.pairs || []).filter(p => p.model === state.model && p.platform !== state.platform);
  if (!state.config || !others.length) {
    views.appendChild(el("p", "empty",
      "Compare needs a config selected and a second measured platform for this model."));
    return;
  }
  e2eView(views, PAYLOAD, "compare-" + PAYLOAD.platform, PAYLOAD.platform + " · " + esc(state.config),
          "The selected pair's cells.");
  const deltas = [];
  for (const o of others) {
    const p = await fetchPayload(o);
    e2eView(views, p, "compare-" + o.platform, o.platform + " · " + esc(state.config),
            "The same (model, config) measured on " + o.platform + ".");
    const mine = (PAYLOAD.e2e[state.config] || {}).cells || [];
    const theirs = (p.e2e[state.config] || {}).cells || [];
    const bk = new Map(theirs.map(x => [x.concurrency + "/" + x.context, x]));
    for (const x of mine) {
      const y = bk.get(x.concurrency + "/" + x.context);
      if (x.tg != null && y && y.tg != null) deltas.push([o.platform, x.concurrency, x.context, x.tg, y.tg]);
    }
  }
  if (deltas.length) {
    views.appendChild(el("h2", null, "Δ decode tokens/s across platforms"));
    views.appendChild(el("div", "tablewrap", table(
      ["platform", "conc", "ctx", PAYLOAD.platform, "other", "Δ"],
      deltas.map(d => [d[0], d[1], LS_LABEL(d[2]), d[3].toFixed(1), d[4].toFixed(1),
                       "<span class='" + (d[4] >= d[3] ? "better" : "worse") + "'>" +
                       ((d[4] / d[3] - 1) * 100).toFixed(1) + "%</span>"]))));
  }
}

// ── the filters themselves ──────────────────────────────────────────────────

function renderPlatformTabs() {
  const tabs = $("filter-platform");
  tabs.innerHTML = "";
  const plats = ["all", ...new Set((MANIFEST.pairs || []).map(p => p.platform))];
  for (const p of plats) {
    const on = state.platform === p || (!state.platform && p === "all");
    const b = el("button", on ? "on" : null, p === "all" ? "All" : p);
    b.type = "button";
    b.onclick = () => setState({platform: p === "all" ? "" : p, model: "", stage: "", config: ""});
    tabs.appendChild(b);
  }
}

function renderPairPicker() {
  $("filter-model-options").innerHTML = "";
  const want = state.platform || "";
  for (const p of MANIFEST.pairs || []) {
    if (want && p.platform !== want) continue;
    const o = el("option");
    o.value = p.platform + "|" + p.model;
    o.label = p.model + " — " + p.platform;
    $("filter-model-options").appendChild(o);
  }
  $("filter-model").value = LOADED ? LOADED.platform + "|" + LOADED.model : "";
}

function renderConfigChips() {
  const chips = $("filter-config");
  chips.innerHTML = "";
  for (const c of (PAYLOAD ? PAYLOAD.configs : [])) {
    const b = el("button", state.config === c.quant ? "on" : null, c.label);
    b.type = "button";
    b.title = c.why;
    b.onclick = () => setState({config: state.config === c.quant ? "" : c.quant});
    chips.appendChild(b);
  }
  chips.closest(".fgroup").style.display = PAYLOAD && PAYLOAD.configs.length ? "" : "none";
}

function renderStageNav() {
  const nav = $("filter-stage");
  nav.innerHTML = "";
  const measured = new Set(PAYLOAD ? [...Object.keys(PAYLOAD.stages),
                                           ...(Object.keys(PAYLOAD.e2e).length ? ["e2e"] : [])] : []);
  const items = [["", "Hero"], ["e2e", "E2E"]]
    .concat(Object.keys(MANIFEST.stages).map(s => [s, MANIFEST.stages[s].title]));
  for (const [key, label] of items) {
    const b = el("button", state.stage === key ? "on" : null, label);
    b.type = "button";
    b.disabled = key !== "" && !measured.has(key);
    b.onclick = () => setState({stage: key});
    nav.appendChild(b);
  }
}

function renderHostOptions() {
  const sel = $("filter-host"), hosts = new Set();
  for (const doc of Object.values(PAYLOAD ? PAYLOAD.stages : {})) if (doc.host) hosts.add(doc.host);
  const cur = sel.value;
  sel.innerHTML = "<option value=''>All hosts</option>";
  for (const h of [...hosts].sort()) sel.appendChild(new Option(h, h));
  sel.value = hosts.has(cur) ? cur : "";
  sel.closest(".fgroup").style.display = hosts.size ? "" : "none";
}

// ── global search: matches model / stage titles, navigates on pick ──────────

function doSearch(q) {
  const box = $("filter-search-results");
  box.innerHTML = "";
  q = q.trim().toLowerCase();
  if (!q) return;
  const matches = [];
  for (const p of MANIFEST.pairs || []) {
    if (p.model.toLowerCase().includes(q)) {
      matches.push({label: p.model + " — " + p.platform,
                    state: {platform: p.platform, model: p.model, stage: "", config: ""}});
    }
  }
  for (const [s, meta] of Object.entries(MANIFEST.stages)) {
    if (s.includes(q) || meta.title.toLowerCase().includes(q)) {
      matches.push({label: "stage: " + meta.title, state: {stage: s}});
    }
  }
  for (const m of matches.slice(0, 8)) {
    const b = el("button", null, m.label);
    b.type = "button";
    b.onclick = () => {
      box.innerHTML = "";
      $("filter-search").value = "";
      setState(m.state);
    };
    box.appendChild(b);
  }
}

// ── boot ────────────────────────────────────────────────────────────────────

function bindFilters() {
  $("filter-model").addEventListener("change", e => {
    const [platform, model] = e.target.value.split("|");
    if (platform && model) setState({platform, model, config: ""});
  });
  for (const id of ["filter-family", "filter-metric", "filter-host", "filter-age"]) {
    $(id).addEventListener("change", refresh);
  }
  $("filter-compare").addEventListener("change", e => setState({compare: e.target.checked}));
  $("filter-search").addEventListener("input", e => doSearch(e.target.value));
  $("toggle").addEventListener("click", () => {
    document.documentElement.dataset.theme =
      document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    refresh();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", refresh);
  addEventListener("hashchange", () => {
    state = parseHash();
    refresh();
  });
}

function defaultPair() {
  return (MANIFEST && MANIFEST.default) || DEFAULT_PAIR;
}

async function refresh() {
  const fallback = (!state.platform || !state.model) ? defaultPair() : null;
  const platform = state.platform || (fallback && fallback.platform);
  const model = state.model || (fallback && fallback.model);
  const entry = pairEntry(platform, model);
  if (!entry) {
    PAYLOAD = LOADED = null;
    $("tiles").innerHTML = "";
    $("views").innerHTML = "<p class='empty'>No measured pair selected — pick a platform and model above.</p>";
    return;
  }
  PAYLOAD = await fetchPayload(entry);
  LOADED = entry;
  $("filter-compare").checked = !!state.compare;
  metricOptions(state.stage);
  renderPlatformTabs();
  renderPairPicker();
  renderConfigChips();
  renderStageNav();
  renderHostOptions();
  $("tiles").innerHTML = "";
  $("views").innerHTML = "";
  if (state.compare && state.stage) return compareView($("views"));
  render();
}

async function boot() {
  MANIFEST = await (await fetch("manifest.json")).json();
  bindFilters();
  await refresh();
}
boot();
