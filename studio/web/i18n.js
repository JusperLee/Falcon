import staticMessages from './i18n-static.js';
import dynamicMessages from './i18n-dynamic.js';
import errorMessages from './i18n-errors.js';
import audioMessages from './i18n-audio.js';

const messages = { ...staticMessages, ...dynamicMessages, ...errorMessages, ...audioMessages };
const STORAGE_KEY = 'falcon.language';
const supported = new Set(['en', 'zh']);
const translatedAttributes = ['title', 'aria-label', 'aria-valuetext', 'placeholder'];
const listeners = new Set();
let initialized = false;
let language = 'en';
try {
  const saved = globalThis.localStorage?.getItem(STORAGE_KEY);
  if (supported.has(saved)) language = saved;
} catch { /* Private browsing can disable storage; English remains the default. */ }

const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const reverse = new Map(Object.entries(messages).map(([zh, en]) => [en, zh]));
function pattern(template) {
  const names = [];
  const pieces = template.split(/(\{[a-zA-Z][\w]*\})/g).map((part) => {
    if (/^\{[a-zA-Z][\w]*\}$/.test(part)) { names.push(part.slice(1, -1)); return '(.*?)'; }
    return escapeRegExp(part);
  });
  return { names, expression: new RegExp(`^${pieces.join('')}$`, 's') };
}
// Error messages can arrive from a language-neutral worker or Python server.
// Recognize our own templates; never translate arbitrary device names or paths.
const patterns = Object.entries(messages)
  .filter(([zh]) => /\{[a-zA-Z][\w]*\}/.test(zh))
  .sort((a, b) => b[0].replace(/\{[^}]+\}/g, '').length - a[0].replace(/\{[^}]+\}/g, '').length)
  .map(([zh, en]) => ({ zh, en, source: pattern(zh), english: pattern(en) }));

function interpolate(template, params, locale, depth = 0) {
  return template.replace(/\{([a-zA-Z][\w]*)\}/g, (token, name) => {
    if (!Object.prototype.hasOwnProperty.call(params, name)) return token;
    const value = params[name];
    return typeof value === 'string' && depth < 4
      ? translateMessage(value, locale, depth + 1) : String(value ?? '');
  });
}

export function translateMessage(message, locale = language, depth = 0) {
  const value = String(message ?? '');
  if (Object.prototype.hasOwnProperty.call(messages, value)) return locale === 'zh' ? value : messages[value];
  const original = reverse.get(value);
  if (original !== undefined) return locale === 'zh' ? original : value;
  if (depth >= 4) return value;
  for (const item of patterns) {
    for (const matcher of [item.source, item.english]) {
      const match = matcher.expression.exec(value);
      if (!match) continue;
      const params = Object.fromEntries(matcher.names.map((name, i) => [name, match[i + 1]]));
      return interpolate(locale === 'zh' ? item.zh : item.en, params, locale, depth + 1);
    }
  }
  return value;
}

export function t(key, params = {}) {
  const source = String(key ?? '');
  const template = Object.prototype.hasOwnProperty.call(messages, source)
    ? language === 'zh' ? source : messages[source] : translateMessage(source, language);
  return interpolate(template, params, language);
}
export function getLanguage() { return language; }

function readParams(element, attribute) {
  try { return JSON.parse(element.getAttribute(attribute) || '{}'); }
  catch { return {}; }
}

export function setText(element, key, params = {}) {
  element.setAttribute('data-i18n', String(key ?? ''));
  element.setAttribute('data-i18n-params', JSON.stringify(params));
  const value = t(key, params);
  if (element.textContent !== value) element.textContent = value;
}
export function setAttr(element, attribute, key, params = {}) {
  if (!translatedAttributes.includes(attribute)) throw new Error(`Unsupported translated attribute: ${attribute}`);
  element.setAttribute(`data-i18n-${attribute}`, String(key ?? ''));
  element.setAttribute(`data-i18n-${attribute}-params`, JSON.stringify(params));
  const value = t(key, params);
  if (element.getAttribute(attribute) !== value) element.setAttribute(attribute, value);
}

function renderLanguage() {
  if (typeof document === 'undefined') return;
  document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  for (const element of document.querySelectorAll('[data-i18n]')) {
    const value = t(element.getAttribute('data-i18n'), readParams(element, 'data-i18n-params'));
    if (element.textContent !== value) element.textContent = value;
  }
  for (const attribute of translatedAttributes) {
    const key = `data-i18n-${attribute}`;
    for (const element of document.querySelectorAll(`[${key}]`)) {
      const value = t(element.getAttribute(key), readParams(element, `${key}-params`));
      if (element.getAttribute(attribute) !== value) element.setAttribute(attribute, value);
    }
  }
  const select = document.getElementById('languageSelect');
  if (select) select.value = language;
}

export function setLanguage(locale) {
  const next = supported.has(locale) ? locale : 'en';
  const changed = language !== next;
  language = next;
  try { globalThis.localStorage?.setItem(STORAGE_KEY, language); } catch { /* Session still works. */ }
  renderLanguage();
  if (changed) for (const listener of listeners) listener(language);
}
export function onLanguageChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
export function initializeI18n() {
  if (!initialized) {
    initialized = true;
    document.getElementById('languageSelect')?.addEventListener('change', (event) => setLanguage(event.target.value));
  }
  renderLanguage();
}
