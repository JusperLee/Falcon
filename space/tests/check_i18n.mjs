/* Every Chinese string the Space can show has an English translation that keeps its
 * placeholders, and index.html's initial English matches it.
 * Run: node space/tests/check_i18n.mjs <assembled Space directory>
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const build = process.argv[2];
assert.ok(build, 'usage: node check_i18n.mjs <assembled Space directory>');
const { t } = await import(pathToFileURL(path.join(build, 'web', 'i18n.js')).href);
const han = /\p{Script=Han}/u;
const read = (file) => fs.readFileSync(path.join(build, file), 'utf8');
const placeholders = (text) => [...text.matchAll(/\{[a-zA-Z]\w*\}/g)].map((m) => m[0]).sort();
const keys = new Set();
const literal = /'((?:[^'\\\n]|\\.)*)'|"((?:[^"\\\n]|\\.)*)"/g;
for (const file of ['web/app.js', 'clip.py', 'app.py']) {
  for (const match of read(file).matchAll(literal)) {
    const text = match[1] ?? match[2];
    if (han.test(text) && !/^[\s\S]*\n/.test(text)) keys.add(text);
  }
}
const html = read('web/index.html');
const initial = new Map();
for (const match of html.matchAll(/data-i18n(?:-[a-z-]+)?="([^"]+)"/g)) if (han.test(match[1])) keys.add(match[1]);
for (const match of html.matchAll(/<[^>]*\sdata-i18n="([^"]+)"(?:\s+data-i18n-params='([^']*)')?[^>]*>([^<]*)</g)) {
  initial.set(match[1], { text: match[3], params: match[2] ? JSON.parse(match[2]) : {} });
}
let failures = 0;
for (const key of keys) {
  if (key === '中文') continue;
  const english = t(key);
  try {
    assert.ok(!han.test(english), `untranslated: ${key}`);
    assert.deepEqual(placeholders(english), placeholders(key), `placeholders differ: ${key} → ${english}`);
    const shown = initial.get(key);
    if (shown && shown.text.trim()) assert.equal(shown.text, t(key, shown.params), `index.html English differs for: ${key}`);
  } catch (error) { failures++; console.error(error.message); }
}
assert.equal(failures, 0, `${failures} i18n problem(s)`);
console.log(`i18n: ${keys.size} Chinese strings translated`);
