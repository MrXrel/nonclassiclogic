import assert from "node:assert/strict";
import { bindThreshold, parseAlpha, pointStatus } from "../ui/live_clusters.mjs";

assert.equal(parseAlpha("0,43"), .43);
assert.equal(parseAlpha("0.43"), .43);
for (const value of ["", "1,01", "-0.01", "NaN", "0,4.3"]) assert.equal(parseAlpha(value), null);
assert.equal(pointStatus(0, null, 0), "noise");
assert.equal(pointStatus(1, .43, .43), "retained");
assert.equal(pointStatus(1, .429, .43), "rejected");

class Control extends EventTarget {
  setAttribute() {}
}
const slider = new Control(), field = new Control(), previews = [], commits = [];
let invalid = false;
const unbind = bindThreshold(slider, field, {
  initial: 0, preview: v => previews.push(v), commit: v => commits.push(v),
  invalid: v => { invalid = v; },
});
for (const alpha of [.125, .43, .652]) {
  slider.value = String(alpha);
  slider.dispatchEvent(new Event("input"));
  assert.equal(previews.at(-1), alpha);
  assert.equal(commits.length, 0);
}
slider.dispatchEvent(new Event("change"));
assert.deepEqual(commits, [.652]);
field.value = "0,43";
field.dispatchEvent(new Event("input"));
assert.equal(previews.at(-1), .43);
assert.deepEqual(commits, [.652]);
field.dispatchEvent(new Event("blur"));
assert.deepEqual(commits, [.652, .43]);
field.value = "3";
field.dispatchEvent(new Event("input"));
assert.equal(invalid, true);
assert.equal(previews.at(-1), .43);
field.dispatchEvent(new Event("blur"));
assert.deepEqual(commits, [.652, .43]);
assert.equal(field.value, "0,430");
unbind();
