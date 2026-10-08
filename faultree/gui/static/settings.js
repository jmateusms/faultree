// Interface preferences (not part of the model), remembered in localStorage
// when available.
//   shared    marker of shared events and cloned gates: "letter" (a letter
//             badge and a colour ring), "color" (colour ring only) or
//             "count" (the number of occurrences, ×n)
//   twins     hovering an occurrence highlights all the others
//   notation  probabilities as "auto" (0.0184, 3.5e-4) or "sci" (1.84e-2)
const KEY = "faultree.settings";
export const CHOICES = { shared: ["letter", "color", "count"], twins: [true, false], notation: ["auto", "sci"] };
const DEFAULTS = { shared: "letter", twins: true, notation: "auto" };
export const SET = { ...DEFAULTS };

export function loadSettings() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch { /* storage blocked or bad JSON */ }
  for (const [k, options] of Object.entries(CHOICES)) if (options.includes(saved[k])) SET[k] = saved[k];
}
export function setSetting(k, v) {
  if (!CHOICES[k] || !CHOICES[k].includes(v)) return;
  SET[k] = v;
  try { localStorage.setItem(KEY, JSON.stringify(SET)); } catch { /* storage blocked */ }
}
export function resetSettings() {
  Object.assign(SET, DEFAULTS);
  try { localStorage.removeItem(KEY); } catch { /* storage blocked */ }
}
