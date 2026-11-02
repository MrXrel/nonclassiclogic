// Pure helpers are exported so numerical and input-event behavior can be tested.
export function membership(x, sigma) {
  if (x <= 0) return 0;
  const t = x / sigma;
  return t * Math.exp((1 - t * t) / 2);
}

export function alphaCut(alpha, sigma) {
  if (!(alpha > 0 && alpha <= 1 && sigma > 0 && Number.isFinite(sigma))) throw new Error("Некорректные параметры среза");
  if (alpha === 1) return [sigma, sigma];
  function bisect(lo, hi, rising) {
    for (let i = 0; i < 70; i++) {
      const mid = (lo + hi) / 2;
      if ((membership(mid, 1) < alpha) === rising) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2 * sigma;
  }
  let hi = 2;
  while (membership(hi, 1) > alpha) hi *= 2;
  return [bisect(0, 1, true), bisect(1, hi, false)];
}

export function bindLiveRange(input, preview, commit) {
  // Input fires throughout a drag; change commits the final value to Python.
  const onInput = () => preview(Number(input.value));
  const onChange = () => commit(Number(input.value));
  input.addEventListener("input", onInput);
  input.addEventListener("change", onChange);
  return () => { input.removeEventListener("input", onInput); input.removeEventListener("change", onChange); };
}

export default function render({ parentElement, data, setStateValue, setTriggerValue }) {
  const root = parentElement.querySelector(".model");
  const sigmaInput = root.querySelector("#sigma"), alphaInput = root.querySelector("#alpha");
  const svg = root.querySelector("svg"), plot = root.querySelector(".plot"), tooltip = root.querySelector(".tooltip");
  const add = root.querySelector("#add");
  const NS = "http://www.w3.org/2000/svg";
  let sigma = Number(data.sigma), alpha = Number(data.alpha), frame = 0, metrics;
  const fmt = value => (Math.abs(value) >= 1e5 || (value !== 0 && Math.abs(value) < 1e-4)
    ? value.toExponential(2) : Number(value.toPrecision(6)).toString()).replace(".", ",");
  sigmaInput.value = sigma;
  alphaInput.min = data.epsilon;
  alphaInput.value = alpha;
  root.querySelector("#alpha-min").textContent = fmt(data.epsilon);
  add.disabled = !data.can_add;

  function node(tag, attrs, parent = svg, text = null) {
    const el = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    if (text !== null) el.textContent = text;
    parent.appendChild(el);
    return el;
  }
  let cutLine, leftPoint, rightPoint, band, guide;
  function drawBase() {
    const width = Math.max(260, plot.getBoundingClientRect().width);
    const height = width < 580 ? 340 : 400;
    svg.style.height = height + "px";
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren();
    const margin = { left: 60, right: 20, top: 28, bottom: 52 };
    const tail = alphaCut(data.epsilon, sigma)[1] * 1.06;
    const rawStep = tail / 5, power = 10 ** Math.floor(Math.log10(rawStep));
    const step = [1, 2, 2.5, 5, 10].find(v => v * power >= rawStep) * power;
    const xmax = Math.ceil(tail / step) * step;
    const x = v => margin.left + v / xmax * (width - margin.left - margin.right);
    const y = v => height - margin.bottom - v / 1.12 * (height - margin.top - margin.bottom);
    metrics = { width, height, margin, x, y, xmax };
    for (let i = 0; i <= 5; i++) {
      const value = i / 5;
      node("line", { x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value), stroke: "#e9edf1" });
      node("text", { x: margin.left - 12, y: y(value) + 4, "text-anchor": "end" }, svg, fmt(value));
    }
    for (let i = 0; i <= Math.round(xmax / step); i++) {
      const v = i * step;
      node("text", { x: x(v), y: height - margin.bottom + 23, "text-anchor": "middle" }, svg, fmt(v));
    }
    node("text", { x: (margin.left + width - margin.right) / 2, y: height - 9, "text-anchor": "middle" }, svg, "Значение x");
    node("text", { x: 17, y: height / 2, transform: `rotate(-90 17 ${height / 2})`, "text-anchor": "middle" }, svg, "Принадлежность μ(x)");
    const values = Array.from({ length: 501 }, (_, i) => i / 500 * xmax).concat(sigma).sort((a, b) => a - b);
    const curve = values.map((v, i) => `${i ? "L" : "M"}${x(v)},${y(membership(v, sigma))}`).join(" ");
    node("path", { d: `${curve} L${x(xmax)},${y(0)} L${x(0)},${y(0)} Z`, fill: "#087f8c", "fill-opacity": ".07" });
    band = node("rect", { fill: "#d47b35", "fill-opacity": ".06", y: margin.top, height: height - margin.top - margin.bottom });
    node("path", { d: curve, stroke: "#087f8c", "stroke-width": "3", fill: "none" });
    guide = node("line", { x1: margin.left, x2: width - margin.right, stroke: "#a0adb9", "stroke-dasharray": "3 4" });
    cutLine = node("line", { stroke: "#d47b35", "stroke-width": "5", "stroke-linecap": "round" });
    leftPoint = node("circle", { r: 4.5, fill: "#d47b35" });
    rightPoint = node("circle", { r: 4.5, fill: "#d47b35" });
    node("text", { x: x(sigma) + 8, y: y(1) - 14, fill: "#087f8c" }, svg, "x = σ");
    updateCut();
  }
  function updateCut() {
    if (!metrics) return;
    const [lower, upper] = alphaCut(alpha, sigma), { x, y, margin, height } = metrics;
    root.dataset.alpha = alpha;
    root.dataset.lower = lower;
    root.dataset.upper = upper;
    root.querySelector("#alpha-value").textContent = alpha.toFixed(3).replace(".", ",");
    root.querySelector("#lower").textContent = fmt(lower);
    root.querySelector("#upper").textContent = fmt(upper);
    root.querySelector("#width").textContent = fmt(upper - lower);
    svg.setAttribute("aria-label", `Рэлей, σ = ${fmt(sigma)}; α = ${fmt(alpha)}; срез от ${fmt(lower)} до ${fmt(upper)}`);
    for (const line of [guide, cutLine]) { line.setAttribute("y1", y(alpha)); line.setAttribute("y2", y(alpha)); }
    cutLine.setAttribute("x1", x(lower)); cutLine.setAttribute("x2", x(upper));
    leftPoint.setAttribute("cx", x(lower)); rightPoint.setAttribute("cx", x(upper));
    for (const point of [leftPoint, rightPoint]) point.setAttribute("cy", y(alpha));
    band.setAttribute("x", x(lower)); band.setAttribute("width", Math.max(0, x(upper) - x(lower)));
    band.setAttribute("height", height - margin.top - margin.bottom);
  }
  function schedule() {
    if (!frame) frame = requestAnimationFrame(() => { frame = 0; updateCut(); });
  }
  function payload() { return { sigma, alpha }; }
  function commit() { updateCut(); setStateValue("controls", payload()); }
  const unbindRange = bindLiveRange(alphaInput, value => { alpha = value; schedule(); }, () => commit());
  function sigmaPreview() {
    if (sigmaInput.validity.valid && Number(sigmaInput.value) > 0) {
      sigma = Number(sigmaInput.value); drawBase();
    }
  }
  function sigmaCommit() {
    if (!sigmaInput.validity.valid) sigmaInput.value = sigma;
    commit();
  }
  function onAdd() { updateCut(); setTriggerValue("add", { ...payload(), request_id: crypto.randomUUID() }); }
  function hover(event) {
    const box = svg.getBoundingClientRect(), { width, height, x, y, xmax, margin } = metrics;
    const px = (event.clientX - box.left) / box.width * width;
    const v = Math.max(0, Math.min(xmax, (px - margin.left) / (width - margin.left - margin.right) * xmax));
    tooltip.hidden = false;
    tooltip.textContent = `x = ${fmt(v)} · μ = ${fmt(membership(v, sigma))}`;
    tooltip.style.left = Math.max(0, Math.min(box.width - tooltip.offsetWidth, px + 12)) + "px";
    tooltip.style.top = Math.max(0, Math.min(height - 35, y(membership(v, sigma)) - 42)) + "px";
  }
  function hideHover() { tooltip.hidden = true; }
  sigmaInput.addEventListener("input", sigmaPreview); sigmaInput.addEventListener("change", sigmaCommit);
  add.addEventListener("click", onAdd); svg.addEventListener("pointermove", hover); svg.addEventListener("pointerleave", hideHover);
  const observer = new ResizeObserver(drawBase); observer.observe(plot); drawBase();
  return () => {
    observer.disconnect(); if (frame) cancelAnimationFrame(frame); unbindRange();
    sigmaInput.removeEventListener("input", sigmaPreview); sigmaInput.removeEventListener("change", sigmaCommit);
    add.removeEventListener("click", onAdd); svg.removeEventListener("pointermove", hover); svg.removeEventListener("pointerleave", hideHover);
  };
}
