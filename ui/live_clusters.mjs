const SVG = "http://www.w3.org/2000/svg";
const fmt = n => Number(n.toPrecision(5)).toString().replace(".", ",");
const alphaText = n => n.toFixed(3).replace(".", ",");
const gray = "#69737e";

export function parseAlpha(text) {
  const value = text.trim().replace(",", ".");
  if (!/^(?:\d+(?:\.\d*)?|\.\d+)$/.test(value)) return null;
  const n = Number(value);
  return n >= 0 && n <= 1 ? Math.round(n * 1000) / 1000 : null;
}

export function pointStatus(label, score, alpha) {
  return label === 0 ? "noise" : score >= alpha ? "retained" : "rejected";
}

export function bindThreshold(slider, field, { initial, preview, commit, invalid }) {
  let alpha = initial, committed = initial;
  const set = (value, syncField) => {
    alpha = value;
    slider.value = value;
    slider.setAttribute("aria-valuetext", alphaText(value));
    if (syncField) field.value = alphaText(value);
    invalid(false);
    preview(value);
  };
  const save = () => {
    if (alpha === committed) return;
    committed = alpha;
    commit(alpha);
  };
  const rangeInput = () => set(Number(slider.value), true);
  const rangeChange = () => { rangeInput(); save(); };
  const textInput = () => {
    const value = parseAlpha(field.value);
    if (value === null) { invalid(true); return; }
    set(value, false);
  };
  const textChange = () => {
    const value = parseAlpha(field.value);
    if (value === null) {
      // Commit the last valid preview, never the malformed value.
      field.value = alphaText(alpha);
      invalid(false);
    } else set(value, true);
    save();
  };
  const key = event => {
    if (event.key === "Enter") { event.preventDefault(); textChange(); }
  };
  const handlers = [[slider, "input", rangeInput], [slider, "change", rangeChange],
    [field, "input", textInput], [field, "blur", textChange], [field, "keydown", key]];
  handlers.forEach(([el, type, fn]) => el.addEventListener(type, fn));
  set(initial, true);
  return () => handlers.forEach(([el, type, fn]) => el.removeEventListener(type, fn));
}

function svgNode(parent, tag, attrs, text) {
  const el = document.createElementNS(SVG, tag);
  Object.entries(attrs).forEach(([name, value]) => el.setAttribute(name, value));
  if (text !== undefined) el.textContent = text;
  parent.append(el);
  return el;
}

function node(parent, tag, text) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  parent.append(el);
  return el;
}

function svgSize(svg, width, height) {
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.style.height = height + "px";
}

function curveMu(x, sigma) {
  const t = x / sigma;
  return t * Math.exp((1 - t * t) / 2);
}

export default function render({ parentElement, data, setStateValue }) {
  const root = parentElement.querySelector(".cluster-filter");
  const methodsBox = root.querySelector(".filter-methods");
  const details = root.querySelector(".filter-inspect");
  const methodSelect = root.querySelector("#cluster-filter-method");
  const groupSelect = root.querySelector("#cluster-filter-group");
  const tbody = root.querySelector("tbody");
  const curve = root.querySelector(".filter-curves");
  const curveSummary = root.querySelector(".filter-curve-summary");
  const field = root.querySelector("#cluster-threshold-text");
  const error = root.querySelector("#filter-error");
  const previous = root.dataset.revision === String(data.revision);
  const savedMethod = previous ? Number(methodSelect.value) : 0;
  const savedGroup = previous ? groupSelect.value : "";
  root.dataset.revision = data.revision;
  if (!previous) details.open = false;
  root.querySelector(".noise-legend").hidden = data.methods.length < 2;
  root.querySelector(".method-select").hidden = data.methods.length < 2;
  let alpha = data.alpha, frame = 0, curves = [], rows = new Map();
  const sigmas = data.items.map(i => i.sigma);
  const low = Math.min(...sigmas), high = Math.max(...sigmas);
  const pad = Math.max((high - low) * .04, high * .02, 1e-9);
  const xMin = Math.max(0, low - pad), xMax = high + pad;

  methodsBox.replaceChildren();
  methodSelect.replaceChildren();
  const panels = data.methods.map((method, index) => {
    const option = node(methodSelect, "option", method.name);
    option.value = index;
    const groups = method.groups.map(g => ({
      ...g, idx: method.labels.flatMap((label, i) => label === g.id ? [i] : []),
    }));
    const block = node(methodsBox, "section");
    block.className = "filter-method";
    const header = node(block, "header");
    node(header, "strong", method.name);
    const summary = node(header, "output");
    const lanes = node(block, "div");
    lanes.className = "filter-lanes";
    if (groups.length > 10) {
      lanes.tabIndex = 0;
      lanes.setAttribute("aria-label", `${method.name}: прокрутка групп`);
    }
    const graph = svgNode(lanes, "svg", { role: "img", "aria-label": `Группы: ${method.name}` });
    const axisWrap = node(block, "div");
    axisWrap.className = "filter-axis";
    const axis = svgNode(axisWrap, "svg", { "aria-hidden": "true" });
    return { ...method, groups, summary, graph, axis, lanes, points: [], counts: new Map() };
  });

  function drawStrips() {
    for (const p of panels) {
      const w = Math.max(230, p.lanes.clientWidth), left = 76, right = 67;
      const x = sigma => left + (sigma - xMin) / (xMax - xMin) * (w - left - right);
      const height = p.groups.length * 25 + 8;
      svgSize(p.graph, w, height);
      svgSize(p.axis, w, 37);
      p.graph.replaceChildren();
      p.axis.replaceChildren();
      p.points = [];
      p.counts.clear();
      const ticks = w < 480 ? 4 : 6;
      for (let j = 0; j < ticks; j++) {
        const value = xMin + (xMax - xMin) * j / (ticks - 1), xx = x(value);
        svgNode(p.graph, "line", { x1: xx, x2: xx, y1: 0, y2: height, class: "filter-grid" });
        svgNode(p.axis, "text", { x: xx, y: 12, "text-anchor": "middle" }, fmt(value));
      }
      svgNode(p.axis, "text", { x: (left + w - right) / 2, y: 30, "text-anchor": "middle" }, "Параметр σ");
      p.groups.forEach((g, j) => {
        const y = j * 25 + 16;
        svgNode(p.graph, "text", { x: 0, y: y + 4 }, g.name);
        svgNode(p.graph, "line", { x1: left, x2: w - right, y1: y, y2: y, class: "filter-grid" });
        if (g.center !== null) {
          const xx = x(g.center);
          const line = svgNode(p.graph, "line", { x1: xx, x2: xx, y1: y - 10, y2: y + 10,
            class: "filter-center", stroke: g.color });
          svgNode(line, "title", {}, `Центр: σ = ${fmt(g.center)}`);
        }
        p.counts.set(g.id, svgNode(p.graph, "text", { x: w - 3, y: y + 4, "text-anchor": "end" }));
        g.idx.forEach(i => {
          const xx = x(sigmas[i]);
          const mark = g.id === 0
            ? svgNode(p.graph, "path", { d: `M${xx-3},${y-3}l6,6m0,-6l-6,6`, class: "filter-noise" })
            : svgNode(p.graph, "circle", { cx: xx, cy: y, r: 4, class: "filter-point",
                style: `--group-color:${g.color}` });
          p.points.push({ i, mark, title: svgNode(mark, "title", {}) });
        });
      });
    }
    paint();
  }

  function selectedPanel() { return panels[Number(methodSelect.value)] || panels[0]; }
  function drawCurves() {
    curves = [];
    if (!details.open) return;
    const p = selectedPanel(), g = p.groups.find(g => String(g.id) === groupSelect.value);
    if (!g) return;
    const w = Math.max(230, curve.clientWidth), h = 236;
    const xmax = Math.max(...g.idx.map(i => sigmas[i])) * data.tail;
    const x = v => 39 + v / xmax * (w - 61), y = u => 195 - u * 172;
    svgSize(curve, w, h);
    curve.replaceChildren();
    for (let i = 0; i <= 4; i++) {
      const xx = x(xmax * i / 4), yy = y(i / 4);
      svgNode(curve, "line", { x1: 39, x2: w - 22, y1: yy, y2: yy, class: "filter-grid" });
      svgNode(curve, "text", { x: 33, y: yy + 4, "text-anchor": "end" }, fmt(i / 4));
      svgNode(curve, "text", { x: xx, y: 213, "text-anchor": "middle" }, fmt(xmax * i / 4));
    }
    svgNode(curve, "text", { x: 12, y: 14 }, "μ(x)");
    svgNode(curve, "text", { x: w / 2, y: 232, "text-anchor": "middle" }, "Значение x");
    g.idx.slice(0, 50).forEach(i => {
      const sigma = sigmas[i];
      const xs = Array.from({ length: 181 }, (_, j) => sigma * data.tail * j / 180);
      xs.push(sigma);
      xs.sort((a, b) => a - b);
      const path = xs.map((v, j) => `${j ? "L" : "M"}${x(v).toFixed(2)},${y(curveMu(v, sigma)).toFixed(2)}`).join(" ");
      const mark = svgNode(curve, "path", { d: path, class: "filter-curve",
        style: `--group-color:${g.id ? g.color : gray}` });
      svgNode(mark, "title", {}, data.items[i].id);
      curves.push({ i, mark });
    });
    paint();
  }

  function setupInspection(group = "") {
    const p = selectedPanel();
    groupSelect.replaceChildren();
    tbody.replaceChildren();
    rows = new Map();
    for (const g of p.groups) {
      const option = node(groupSelect, "option", g.name);
      option.value = g.id;
      const tr = node(tbody, "tr");
      node(tr, "td", g.name);
      node(tr, "td", g.idx.length);
      const retained = node(tr, "td");
      const rejected = node(tr, "td");
      node(tr, "td", g.center === null ? "—" : fmt(g.center));
      rows.set(g.id, { retained, rejected });
    }
    groupSelect.value = p.groups.some(g => String(g.id) === group) ? group : p.groups[0].id;
    drawCurves();
    paint();
  }

  function paint() {
    for (const p of panels) {
      let kept = 0, assigned = 0, noise = 0;
      for (const g of p.groups) {
        const count = g.id ? g.idx.filter(i => p.scores[i] >= alpha).length : 0;
        g.kept = count;
        kept += count;
        assigned += g.id ? g.idx.length : 0;
        noise += g.id ? 0 : g.idx.length;
        const label = p.counts.get(g.id);
        if (label) {
          label.textContent = g.id ? `${count} из ${g.idx.length}` : String(noise);
          label.setAttribute("aria-label", g.id ? `${g.name}: осталось ${count} из ${g.idx.length}` : `Шум: ${noise}`);
        }
      }
      p.summary.textContent = `Осталось ${kept} из ${assigned} · отсеяно ${assigned-kept}${noise ? ` · шум ${noise}` : ""}`;
      for (const { i, mark, title } of p.points) {
        const status = pointStatus(p.labels[i], p.scores[i], alpha);
        if (mark.dataset.status === status) continue;
        mark.dataset.status = status;
        title.textContent = `${data.items[i].id} · σ = ${fmt(sigmas[i])} · ${status === "noise" ? "шум"
          : `оценка ${fmt(p.scores[i])} · ${status === "retained" ? "прошла порог" : "ниже порога"}`}`;
      }
    }
    const p = selectedPanel();
    for (const g of p.groups) {
      const row = rows.get(g.id);
      if (row) {
        row.retained.textContent = g.id ? g.kept : "—";
        row.rejected.textContent = g.id ? g.idx.length - g.kept : "—";
      }
    }
    const g = p.groups.find(g => String(g.id) === groupSelect.value);
    if (g) {
      curveSummary.textContent = `${g.name} · ${g.id ? `осталось ${g.kept} из ${g.idx.length}` : `${g.idx.length} величин`} · показано до 50 кривых`;
    }
    for (const { i, mark } of curves) mark.dataset.status = pointStatus(p.labels[i], p.scores[i], alpha);
    root.dataset.alpha = alpha;
  }

  const resize = new ResizeObserver(() => { drawStrips(); drawCurves(); });
  panels.forEach(p => resize.observe(p.lanes));
  methodSelect.value = panels[savedMethod] ? savedMethod : 0;
  setupInspection(savedGroup);
  drawStrips();
  const onMethod = () => setupInspection(), onGroup = () => drawCurves();
  const onDetails = () => drawCurves();
  methodSelect.addEventListener("change", onMethod);
  groupSelect.addEventListener("change", onGroup);
  details.addEventListener("toggle", onDetails);
  const unbind = bindThreshold(root.querySelector("#cluster-threshold"), field, {
    initial: alpha,
    preview: value => {
      alpha = value;
      if (!frame) frame = requestAnimationFrame(() => { frame = 0; paint(); });
    },
    commit: value => setStateValue("threshold", { alpha: value, revision: data.revision }),
    invalid: invalid => {
      field.setAttribute("aria-invalid", invalid);
      error.hidden = !invalid;
    },
  });
  return () => {
    unbind();
    resize.disconnect();
    if (frame) cancelAnimationFrame(frame);
    methodSelect.removeEventListener("change", onMethod);
    groupSelect.removeEventListener("change", onGroup);
    details.removeEventListener("toggle", onDetails);
  };
}
