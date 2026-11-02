import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {alphaCut, membership, bindLiveRange} from '../ui/live_model.mjs';

const references = JSON.parse(readFileSync(0, 'utf8'));
for (const {alpha,sigma,left,right} of references) {
  const bounds=alphaCut(alpha,sigma);
  for (let i=0;i<2;i++) assert.ok(Math.abs(bounds[i]-[left,right][i])<1e-10*Math.max(1,sigma));
  assert.ok(Math.abs(membership(bounds[0],sigma)-alpha)<1e-10);
  assert.ok(Math.abs(membership(bounds[1],sigma)-alpha)<1e-10);
}

class Input extends EventTarget {value='0.5';}
const input=new Input(), previews=[], commits=[];
const unbind=bindLiveRange(input,v=>previews.push(v),v=>commits.push(v));
input.value='0.8';input.dispatchEvent(new Event('input'));
assert.deepEqual(previews,[.8]);assert.deepEqual(commits,[]);
input.value='0.9';input.dispatchEvent(new Event('input'));
assert.deepEqual(previews,[.8,.9]);assert.deepEqual(commits,[]);
input.dispatchEvent(new Event('change'));
assert.deepEqual(commits,[.9]);
unbind();input.value='0.7';input.dispatchEvent(new Event('input'));
assert.deepEqual(previews,[.8,.9]);
console.log('Live preview updates before change/release; numeric roots agree with Python.');
