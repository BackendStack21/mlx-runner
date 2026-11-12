// mlx-runner.21no.de: live release data, the token-stream replay, the terminal, the app tour.
//
// Nothing about a release is baked into the page: the latest release (version, file names, sizes,
// SHA-256 digests) is read from GitHub's API here, and the last one seen is kept for repeat visits.
// Without JavaScript, or if neither is available, every download link already points at the GitHub
// "latest release" page.
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
    for (const el of document.querySelectorAll(".meter, .pipeline, .carousel, .term, .server-grid, .dl-card")) {
      const r = el.getBoundingClientRect();
      if (r.top < innerHeight) continue; // already on screen: leave it alone
      el.classList.add("reveal");
      onVisible(el, () => el.classList.add("in"), 0.12);
    }
  }

  // ── 1. hero: stream an answer at each model's measured decode speed ──
  const MODELS = [
    { name: "Qwen2.5 0.5B Instruct · 4-bit", rate: 343 },
    { name: "Qwen3 0.6B · 4-bit", rate: 286 },
    { name: "Llama 3.2 1B Instruct · 4-bit", rate: 202 },
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

  // ── 2. terminal: replay a captured session ──
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

  // ── 3. the app tour: WAI-ARIA tabs (arrow keys, Home, End) that drive a scroll-snap carousel.
  //      Selecting a tab scrolls to its slide, and swiping or the arrows move the selection. ──
  const carousel = document.querySelector(".carousel");
  if (carousel) {
    const tabs = [...carousel.querySelectorAll('[role="tab"]')];
    const panels = tabs.map((t) => $(t.getAttribute("aria-controls")));
    const slides = carousel.querySelector(".slides");
    let current = Math.max(0, tabs.findIndex((t) => t.getAttribute("aria-selected") === "true"));
    let target = -1; // the slide a click or key is scrolling to: the scroll handler leaves the selection alone until it arrives
    const at = (i) => panels[i].offsetLeft - slides.offsetLeft;
    const mark = (i) => {
      current = i;
      tabs.forEach((t, j) => {
        t.setAttribute("aria-selected", String(i === j));
        t.tabIndex = i === j ? 0 : -1;
      });
    };
    const select = (i, focus) => {
      mark(i);
      target = Math.abs(slides.scrollLeft - at(i)) > 1 ? i : -1; // already there: no scroll will come to clear it
      slides.scrollTo({ left: at(i), behavior: reduced ? "auto" : "smooth" });
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
        else if (e.key === " ") select(i, true); // the tabs are links, which Space does not activate
        else return;
        e.preventDefault();
      });
    });
    mark(current); // one tab stop: the inactive tabs carry no tabindex in the markup, so they work without JS
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
          if (i < 0) return;
          if (target >= 0) {
            if (i === target) target = -1;
            return;
          }
          if (i !== current) mark(i);
        });
      },
      { passive: true },
    );
    // A swipe or a sideways wheel takes over from a scroll that a click started.
    slides.addEventListener("touchstart", () => (target = -1), { passive: true });
    slides.addEventListener("wheel", (e) => Math.abs(e.deltaX) > Math.abs(e.deltaY) && (target = -1), { passive: true });
    slides.addEventListener("scrollend", () => (target = -1));
    carousel.querySelector(".car-btn.prev").addEventListener("click", () => select((current - 1 + panels.length) % panels.length));
    carousel.querySelector(".car-btn.next").addEventListener("click", () => select((current + 1) % panels.length));
  }

  // ── 4. live release data ──
  // The release list is asked first: GitHub's /releases/latest has timed out or answered 504 for
  // minutes at a time while the list answered in under a second. Each request gives up after 8 s.
  // The last release seen is kept in localStorage: a repeat visit shows it at once, and it stands in
  // when GitHub can't be reached or rate-limits the visitor (60 unauthenticated requests an hour).
  const config = JSON.parse($("release-config").textContent);
  const API = `https://api.github.com/repos/${config.repo}/releases`;
  const CACHE = "mlxr:release";
  const FRESH = 10 * 60 * 1000;
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

  // Only the fields the page uses are stored.
  const trim = (rel) => ({
    tag_name: rel.tag_name,
    published_at: rel.published_at,
    assets: (rel.assets || []).map((a) => ({ name: a.name, size: a.size, digest: a.digest, browser_download_url: a.browser_download_url })),
  });

  async function get(url) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 8000);
    try {
      const res = await fetch(url, { headers: { accept: "application/vnd.github+json" }, signal: ctl.signal });
      if (res.status === 404) return null;
      if (!res.ok) throw new Error(`GitHub ${res.status}`);
      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }

  // The newest release that is neither a draft nor a prerelease (what /latest returns), or null if
  // none is published yet.
  async function latest() {
    try {
      const list = await get(`${API}?per_page=10`);
      if (Array.isArray(list)) return list.find((r) => !r.draft && !r.prerelease) || null;
    } catch {}
    return get(`${API}/latest`);
  }

  function saved() {
    try {
      const s = JSON.parse(localStorage.getItem(CACHE) || "null");
      if (s && typeof s.at === "number" && s.rel && typeof s.rel.tag_name === "string" && Array.isArray(s.rel.assets)) return s;
    } catch {}
    return null;
  }

  function fill(r) {
    const { dmg } = r.assets;
    if (dmg) {
      $("dl-primary").href = dmg.url;
      $("dl-dmg").href = dmg.url;
      $("dl-dmg").querySelector("span").textContent = `Download ${r.tag} (.dmg, ${fmtSize(dmg.size)})`;
    }
    $("release-line").textContent = `${r.tag} · Apple silicon · macOS 15 or newer`;
    $("release-status").textContent = `Latest: ${r.tag}, released ${fmtDate(r.published)}. Built and signed by GitHub Actions.`;
    for (const tr of document.querySelectorAll("tr[data-asset]")) {
      const a = r.assets[tr.dataset.asset];
      tr.hidden = !a;
      if (!a) continue;
      const link = tr.querySelector("[data-link]");
      link.href = a.url;
      link.textContent = a.name;
      tr.querySelector("[data-size]").textContent = fmtSize(a.size);
      // fill runs twice when a saved release is shown first, so the cell keeps its data-hash marker
      let hash;
      if (/^[0-9a-f]{64}$/.test(a.sha256)) {
        hash = document.createElement("button");
        hash.className = "sha";
        hash.type = "button";
        hash.title = "Copy SHA-256";
        hash.setAttribute("aria-label", `Copy the SHA-256 of ${a.name}`);
        hash.dataset.copy = a.sha256;
        hash.innerHTML = `<code>${a.sha256.slice(0, 10)}…${a.sha256.slice(-6)}</code>${COPY_ICON}`;
      } else {
        hash = document.createElement("span");
        hash.className = "muted";
        hash.textContent = "in SHA256SUMS";
      }
      hash.dataset.hash = "";
      tr.querySelector("[data-hash]").replaceWith(hash);
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

  const last = saved();
  if (last) fill(parse(last.rel));
  if (!last || Date.now() - last.at > FRESH) {
    latest()
      .then((rel) => {
        if (!rel) {
          if (last) return;
          $("release-status").textContent = "No release is published yet. Watch the repository on GitHub, or build it from source.";
          for (const el of document.querySelectorAll("[data-needs]")) el.hidden = true;
          return;
        }
        const t = trim(rel);
        try {
          localStorage.setItem(CACHE, JSON.stringify({ at: Date.now(), rel: t }));
        } catch {}
        fill(parse(t));
      })
      .catch(() => {
        $("release-status").textContent = last
          ? `Showing ${last.rel.tag_name}, saved from an earlier visit: GitHub didn't answer just now, so a newer release may exist.`
          : "Couldn't load the release details right now; the links open the latest release on GitHub.";
      });
  }

  // ── verify: the commands are collapsed, and a link to #verify opens them ──
  const steps = $("verify-steps");
  const openVerify = () => {
    if (steps && location.hash === "#verify") steps.open = true;
  };
  addEventListener("hashchange", openVerify);
  // A link clicked while the hash is already #verify fires no hashchange.
  for (const a of document.querySelectorAll('a[href="#verify"]')) a.addEventListener("click", () => steps && (steps.open = true));
  openVerify();

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
