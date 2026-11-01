// mlx-runner.21no.de: live release data, the token-stream replay, the race, the terminal, tabs.
//
// Nothing about a release is baked into the page: the latest release (version, file names, sizes,
// SHA-256 digests) is read from GitHub's API here. Without JavaScript, or if the API can't be
// reached, every download link already points at the GitHub "latest release" page.
//
// The speed demos replay measured `mlxr bench` numbers (M2 Pro, 16 GB). Nothing runs a model here.
(() => {
  document.documentElement.classList.add("js");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (id) => document.getElementById(id);

  // ── when something scrolls into view ──
  const onVisible = (el, fn, threshold = 0.25) => {
    if (!el) return;
    if (!("IntersectionObserver" in window)) return fn();
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          io.disconnect();
          fn();
        }
      },
      { threshold },
    );
    io.observe(el);
  };

  // Reveal on scroll. Hidden only once JS has tagged them, so no-JS readers see everything.
  if (!reduced) {
    for (const el of document.querySelectorAll(".race-card, .ceiling-copy, .gauge, .pipeline, .cards > li, .carousel, .term, .code-card, .dl-card, .steps > li")) {
      const r = el.getBoundingClientRect();
      if (r.top < innerHeight) continue; // already on screen: leave it alone
      el.classList.add("reveal");
      onVisible(el, () => el.classList.add("in"), 0.12);
    }
  }
  onVisible(document.querySelector(".gauge"), () => document.querySelector(".gauge").classList.add("in"), 0.4);

  // ── 1. hero: stream an answer at each model's measured decode speed ──
  const MODELS = [
    { name: "Qwen2.5 0.5B Instruct · 4-bit", rate: 358 },
    { name: "Qwen3 0.6B · 4-bit", rate: 289 },
    { name: "Llama 3.2 1B Instruct · 4-bit", rate: 203 },
    { name: "Phi-3.5 mini · 4-bit", rate: 70 },
    { name: "Mistral 7B v0.3 · 4-bit", rate: 41.0 },
  ];
  const stream = $("stream");
  if (stream) {
    const text = $("stream-sr").textContent.trim();
    // Rough subword split: long words become several pieces, like a BPE tokenizer would make.
    const tokens = text.match(/\s*[A-Za-z]{1,6}|\s*\d+(?:[.,]\d+)*|\s*[^\sA-Za-z\d]/g) || [text];
    const out = $("stream-out");
    const rateEl = $("stream-rate");
    const countEl = $("stream-count");
    const timeEl = $("stream-time");
    const bar = $("stream-bar");
    const answer = out.parentElement;
    const buttons = [...stream.querySelectorAll("[data-model]")];
    let model = MODELS[0];
    let raf = 0;
    let shown = 0;

    const fmtRate = (r) => (r >= 100 ? Math.round(r).toString() : r.toFixed(1));
    function render(n, seconds, done) {
      if (n !== shown) {
        out.textContent = tokens.slice(0, n).join("");
        shown = n;
        answer.scrollTop = answer.scrollHeight;
      }
      countEl.textContent = `${n} / ${tokens.length}`;
      timeEl.textContent = `${seconds.toFixed(2)} s`;
      bar.style.setProperty("--p", (n / tokens.length).toFixed(4));
      answer.classList.toggle("fit", done);
      stream.classList.toggle("running", !done);
    }
    function play(i) {
      cancelAnimationFrame(raf);
      model = MODELS[i];
      for (const b of buttons) b.setAttribute("aria-pressed", String(b.dataset.model === String(i)));
      $("stream-model").textContent = model.name;
      rateEl.textContent = fmtRate(model.rate);
      shown = -1;
      if (reduced) return render(tokens.length, tokens.length / model.rate, true);
      let t = 0;
      let last = performance.now();
      const step = (now) => {
        t += Math.min(now - last, 100) / 1000; // a backgrounded tab resumes instead of jumping
        last = now;
        const n = Math.min(tokens.length, Math.floor(t * model.rate));
        render(n, n === tokens.length ? tokens.length / model.rate : t, n === tokens.length);
        if (n < tokens.length) raf = requestAnimationFrame(step);
      };
      render(0, 0, false);
      raf = requestAnimationFrame(step);
    }
    for (const b of buttons) b.addEventListener("click", () => play(Number(b.dataset.model)));
    $("stream-replay").addEventListener("click", () => play(MODELS.indexOf(model)));
    play(0);
  }

  // ── 2. the race: 512-token prompt + 256-token answer per model, on a 10 s axis ──
  const lanes = [...document.querySelectorAll("#lanes li")];
  if (lanes.length) {
    const PROMPT = 512;
    const ANSWER = 256;
    const AXIS = 10;
    const clock = $("race-clock");
    const runBtn = $("race-run");
    const spec = lanes.map((li) => {
      const pf = PROMPT / Number(li.dataset.prefill);
      const dc = ANSWER / Number(li.dataset.decode);
      return { li, pf, dc, total: pf + dc, track: li.querySelector(".track"), label: li.querySelector(".lane-t") };
    });
    const longest = Math.max(...spec.map((s) => s.total));
    let raf = 0;
    function draw(t) {
      for (const s of spec) {
        const pf = Math.min(t, s.pf);
        const dc = Math.max(0, Math.min(t, s.total) - s.pf);
        s.track.style.setProperty("--pf", ((pf / AXIS) * 100).toFixed(3));
        s.track.style.setProperty("--dc", ((dc / AXIS) * 100).toFixed(3));
        const done = t >= s.total;
        s.li.classList.toggle("done", done);
        if (done) s.label.textContent = `${s.total.toFixed(2)} s`;
        else if (t < s.pf) s.label.textContent = "prefill";
        else s.label.textContent = `${Math.floor((t - s.pf) * Number(s.li.dataset.decode))} tok`;
      }
      clock.textContent = Math.min(t, longest).toFixed(2);
    }
    function race() {
      cancelAnimationFrame(raf);
      if (reduced) return draw(longest);
      let t = 0;
      let last = performance.now();
      runBtn.textContent = "Racing…";
      const step = (now) => {
        t += Math.min(now - last, 100) / 1000;
        last = now;
        draw(t);
        if (t < longest) raf = requestAnimationFrame(step);
        else runBtn.textContent = "Run it again";
      };
      draw(0);
      raf = requestAnimationFrame(step);
    }
    draw(longest); // the finished chart is the static state (no JS, reduced motion, snapshots)
    runBtn.addEventListener("click", race);
    if (!reduced) onVisible(document.querySelector(".race-card"), race, 0.5);
  }

  // ── 3. terminal: replay a captured session ──
  const term = $("term");
  if (term && !reduced) {
    const code = term.querySelector("code");
    const original = [...code.childNodes].map((n) => n.cloneNode(true));
    // Flatten into runs of { text, cls } split per line, so we can type commands and print output.
    const runs = [];
    for (const n of original) {
      const cls = n.nodeType === 1 ? n.className : "";
      for (const [i, part] of n.textContent.split("\n").entries()) {
        if (i) runs.push({ text: "\n", cls: "" });
        if (part) runs.push({ text: part, cls });
      }
    }
    let timer = 0;
    let gen = 0;
    const sleep = (ms) => new Promise((r) => (timer = setTimeout(r, ms)));
    async function replay() {
      const me = ++gen;
      clearTimeout(timer);
      code.textContent = "";
      const cursor = document.createElement("span");
      cursor.className = "tcursor";
      let line = null;
      let typing = false; // after a "$" prompt, the rest of the line is typed
      for (const r of runs) {
        if (me !== gen) return;
        if (r.text === "\n") {
          code.append("\n");
          line = null;
          if (typing) {
            typing = false;
            code.append(cursor);
            await sleep(420); // the command runs
            cursor.remove();
          } else await sleep(70);
          continue;
        }
        const span = r.cls ? Object.assign(document.createElement("span"), { className: r.cls }) : document.createTextNode("");
        code.append(span);
        line = span;
        if (r.cls === "p") {
          span.textContent = r.text;
          typing = true;
          await sleep(500);
          continue;
        }
        if (typing) {
          for (const ch of r.text) {
            if (me !== gen) return;
            line.textContent += ch;
            code.append(cursor);
            await sleep(ch === " " ? 26 : 14);
          }
        } else {
          line.textContent = r.text;
        }
        term.scrollTop = term.scrollHeight;
      }
      code.append(cursor);
    }
    $("term-replay").addEventListener("click", replay);
    onVisible(term, replay, 0.4);
  }

  // ── 4. tab lists (WAI-ARIA tabs with arrow keys). The app tour's tabs drive a scroll-snap carousel:
  //      selecting one scrolls to its slide, and swiping or the arrows move the selection. ──
  for (const list of document.querySelectorAll('[role="tablist"]')) {
    const tabs = [...list.querySelectorAll('[role="tab"]')];
    const panels = tabs.map((t) => $(t.getAttribute("aria-controls")));
    const carousel = list.closest(".carousel");
    const slides = carousel ? carousel.querySelector(".slides") : null;
    let current = Math.max(0, tabs.findIndex((t) => t.getAttribute("aria-selected") === "true"));
    const mark = (i) => {
      current = i;
      tabs.forEach((t, j) => {
        t.setAttribute("aria-selected", String(i === j));
        t.tabIndex = i === j ? 0 : -1;
        if (!slides) panels[j].hidden = i !== j;
      });
    };
    const select = (i, focus) => {
      mark(i);
      if (slides) slides.scrollTo({ left: panels[i].offsetLeft - slides.offsetLeft, behavior: reduced ? "auto" : "smooth" });
      if (focus) tabs[i].focus();
    };
    tabs.forEach((t, i) => {
      t.addEventListener("click", (e) => {
        e.preventDefault();
        select(i);
      });
      t.addEventListener("keydown", (e) => {
        const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
        if (d) select((i + d + tabs.length) % tabs.length, true);
        else if (e.key === "Home") select(0, true);
        else if (e.key === "End") select(tabs.length - 1, true);
        else return;
        e.preventDefault();
      });
    });
    if (!slides) continue;
    let pending = false;
    slides.addEventListener(
      "scroll",
      () => {
        if (pending) return;
        pending = true;
        requestAnimationFrame(() => {
          pending = false;
          const x = slides.scrollLeft + slides.offsetLeft;
          const i = panels.findIndex((p) => Math.abs(p.offsetLeft - x) < p.offsetWidth / 2);
          if (i >= 0 && i !== current) mark(i);
        });
      },
      { passive: true },
    );
    carousel.querySelector(".car-btn.prev").addEventListener("click", () => select((current - 1 + panels.length) % panels.length));
    carousel.querySelector(".car-btn.next").addEventListener("click", () => select((current + 1) % panels.length));
  }

  // ── 5. live release data ──
  const config = JSON.parse($("release-config").textContent);
  const API = `https://api.github.com/repos/${config.repo}/releases/latest`;
  const CACHE = "mlxr:latest-release";
  const COPY_ICON = document.querySelector(".copy svg") ? document.querySelector(".copy svg").outerHTML : "";
  const fmtSize = (n) => (n >= 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(n >= 100 * 1024 * 1024 ? 0 : 1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);
  const fmtDate = (iso) => new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });

  function parse(rel) {
    const assets = {};
    for (const a of rel.assets || []) {
      const c = config.assets.find((x) => (x.name ? a.name === x.name : a.name.endsWith(x.suffix)) && !assets[x.id]);
      if (c) assets[c.id] = { name: a.name, url: a.browser_download_url, size: a.size, sha256: (a.digest || "").replace(/^sha256:/, "") };
    }
    return { tag: rel.tag_name, published: rel.published_at, assets, names: new Set((rel.assets || []).map((a) => a.name)) };
  }

  async function load() {
    try {
      const cached = JSON.parse(sessionStorage.getItem(CACHE) || "null");
      if (cached && Date.now() - cached.at < 10 * 60 * 1000) return cached.rel;
    } catch {}
    const res = await fetch(API, { headers: { accept: "application/vnd.github+json" } });
    if (res.status === 404) return null; // no release published yet
    if (!res.ok) throw new Error(`GitHub ${res.status}`);
    const rel = await res.json();
    try {
      sessionStorage.setItem(CACHE, JSON.stringify({ at: Date.now(), rel }));
    } catch {}
    return rel;
  }

  function fill(r) {
    const { dmg, cli } = r.assets;
    if (dmg) {
      $("dl-primary").href = dmg.url;
      $("dl-dmg").href = dmg.url;
      $("dl-dmg").querySelector("span").textContent = `Download ${r.tag} (.dmg, ${fmtSize(dmg.size)})`;
    }
    if (cli) $("dl-cli").href = cli.url;
    $("release-line").textContent = `${r.tag} · Apple silicon · macOS 15 or newer`;
    $("release-status").textContent = `Latest: ${r.tag}, released ${fmtDate(r.published)}. Built and signed by GitHub Actions.`;
    for (const tr of document.querySelectorAll("tr[data-asset]")) {
      const a = r.assets[tr.dataset.asset];
      if (!a) {
        tr.hidden = true;
        continue;
      }
      const link = tr.querySelector("[data-link]");
      link.href = a.url;
      link.textContent = a.name;
      tr.querySelector("[data-size]").textContent = fmtSize(a.size);
      const hash = tr.querySelector("[data-hash]");
      if (a.sha256) {
        const b = document.createElement("button");
        b.className = "sha";
        b.type = "button";
        b.title = "Copy SHA-256";
        b.setAttribute("aria-label", `Copy the SHA-256 of ${a.name}`);
        b.dataset.copy = a.sha256;
        b.innerHTML = `<code>${a.sha256.slice(0, 10)}…${a.sha256.slice(-6)}</code>${COPY_ICON}`;
        hash.replaceWith(b);
      } else hash.textContent = "in SHA256SUMS";
    }
    for (const el of document.querySelectorAll("[data-name]")) {
      const a = r.assets[el.dataset.name];
      if (a) el.textContent = a.name;
    }
    for (const box of document.querySelectorAll("[data-template]")) {
      let missing = false;
      const asset = (id) => r.assets[id] || ((missing = true), { url: "", name: "" });
      const text = box.dataset.template
        .replace(/\{url:(\w+)\}/g, (_, id) => asset(id).url)
        .replace(/\{name:(\w+)\}/g, (_, id) => asset(id).name)
        .replace(/\{dir:(\w+)\}/g, (_, id) => asset(id).name.replace(/\.tar\.gz$/, ""))
        .replace(/\{tag\}/g, r.tag);
      if (missing) continue;
      const lines = text.split("\n");
      const codeEl = box.querySelector("code");
      codeEl.textContent = "";
      lines.forEach((l, i) => {
        if (i) codeEl.append("\n");
        const hashAt = l.indexOf("#");
        if (hashAt === 0 || / #/.test(l)) {
          const at = hashAt === 0 ? 0 : l.indexOf(" #");
          codeEl.append(l.slice(0, at));
          const c = document.createElement("span");
          c.className = "c";
          c.textContent = l.slice(at);
          codeEl.append(c);
        } else codeEl.append(l);
      });
      box.querySelector(".copy").dataset.copy = lines
        .filter((l) => !l.startsWith("#"))
        .map((l) => l.replace(/\s+#.*$/, ""))
        .join("\n");
    }
    // Steps that need an asset this release doesn't have (e.g. an unsigned release)
    for (const el of document.querySelectorAll("[data-needs]")) el.hidden = !r.names.has(el.dataset.needs);
  }

  load()
    .then((rel) => {
      if (!rel) {
        $("release-status").textContent = "No release is published yet. Watch the repository on GitHub, or build it from source below.";
        for (const el of document.querySelectorAll("[data-needs]")) el.hidden = true;
        return;
      }
      fill(parse(rel));
    })
    .catch(() => {
      $("release-status").textContent = "Couldn't load the release details right now; the links open the latest release on GitHub.";
    });

  // ── copy buttons ──
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-copy]");
    if (!b || !navigator.clipboard) return;
    navigator.clipboard.writeText(b.dataset.copy).then(() => {
      b.classList.add("copied");
      setTimeout(() => b.classList.remove("copied"), 1400);
    });
  });
})();
