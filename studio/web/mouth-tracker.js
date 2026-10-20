/* Pure geometry: moving square crops, not affine alignment or face recognition. */
export const OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
  291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
// Tunable recovery behavior; values will be adjusted later from measurements.
export const defaultPolicy = {
  graceMs: 500, // misses within this do not change status or reset the window; ROI coasts.
  relockFrames: 4, // consecutive consistent frames required to auto re-lock after a loss.
  inconsistentRelockMs: 2500, // stable period on a jumped candidate before auto-locking it.
  shiftLimit: .75, // fraction of face width; beyond this a candidate is "inconsistent".
  scaleLimit: 1.8, // face-width ratio bound in either direction for "consistent".
};
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const validPoint = (p) => p && Number.isFinite(p.x) && Number.isFinite(p.y)
  && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;
const distance = (a, b, w, h) => Math.hypot((a.x - b.x) * w, (a.y - b.y) * h);

export function newMouthTrack() {
  return { roi: null, face: null, lastSeenMs: null, scaleRatio: null, lostSinceMs: null,
    priorTarget: null, holdRoi: null, candidateFace: null, candidateStreak: 0, candidateStableMs: null };
}
// Once a target has been seen, a missing face keeps the last crop so separation never pauses.
function lost(state, { message = '未检测到清晰嘴部，请正对摄像头。', ...fields } = {}) {
  const next = { ...newMouthTrack(), lostSinceMs: state.lostSinceMs, priorTarget: state.priorTarget,
    holdRoi: state.holdRoi, ...fields };
  if (next.holdRoi) return { state: next, status: 'tracking', roi: next.holdRoi, lips: [], coasted: true, searching: true, message };
  return { state: next, status: 'lost', message };
}

export function clampROI(roi, width, height) {
  const size = clamp(Number(roi?.size) || .20, .06, .75);
  const side = Math.min(width, height) * size;
  return {
    cx: clamp(Number.isFinite(roi?.cx) ? roi.cx : .5, side / width / 2, 1 - side / width / 2),
    cy: clamp(Number.isFinite(roi?.cy) ? roi.cy : .6, side / height / 2, 1 - side / height / 2), size,
  };
}

export function mouthGeometry(landmarks, width, height) {
  if (!Array.isArray(landmarks) || !Number.isFinite(width) || !Number.isFinite(height)
      || width < 1 || height < 1) return null;
  if ([61, 291, 234, 454, 1, ...OUTER_LIP_INDICES].some((i) => !validPoint(landmarks[i]))) return null;
  const left = landmarks[61], right = landmarks[291];
  const faceWidth = distance(landmarks[234], landmarks[454], width, height);
  const mouthWidth = distance(left, right, width, height);
  if (faceWidth < 35 || mouthWidth < 8 || mouthWidth > faceWidth * .85) return null;
  // The supplied mean face has a 45-pixel mouth in a 96-pixel crop (ratio 2.13).
  // Calibrate scale once; subsequent scale follows face width, not lip opening.
  return {
    center: { x: (left.x + right.x) / 2, y: (left.y + right.y) / 2 },
    face: { x: landmarks[1].x, y: landmarks[1].y, width: faceWidth },
    scaleRatio: clamp(2.13 * mouthWidth / faceWidth, .45, 1.1),
    lips: OUTER_LIP_INDICES.map((i) => ({ x: landmarks[i].x, y: landmarks[i].y })),
  };
}

// Geometric continuity with the last locked target. Cannot identify people at the
// same position; deliberately not presented as face recognition.
function consistent(face, target, width, height, policy) {
  const shift = distance(face, target, width, height) / target.width;
  const scaleChange = face.width / target.width;
  return shift <= policy.shiftLimit && scaleChange <= policy.scaleLimit && scaleChange >= 1 / policy.scaleLimit;
}
// Nearest consistent face, so a second person close by never wins over the target.
function nearestConsistent(faces, target, width, height, policy) {
  let best = null, bestDistance = Infinity;
  for (const face of faces) {
    const d = distance(face.face, target, width, height);
    if (d < bestDistance && consistent(face.face, target, width, height, policy)) { best = face; bestDistance = d; }
  }
  return best;
}
function cropROI(geometry, scaleRatio, width, height) {
  return clampROI({ cx: geometry.center.x, cy: geometry.center.y,
    size: geometry.face.width * scaleRatio / Math.min(width, height) }, width, height);
}
function lockOn(geometry, roi, nowMs, scaleRatio) {
  return { roi, face: geometry.face, lastSeenMs: nowMs, scaleRatio, lostSinceMs: null,
    priorTarget: null, holdRoi: roi, candidateFace: null, candidateStreak: 0, candidateStableMs: null };
}

// landmarksList: every face this frame detected (usually 0 or 1; briefly more while
// re-acquiring). state.face / priorTarget anchor continuity; neither is an identity.
export function updateMouthTrack(previous, landmarksList, width, height, nowMs, policy = defaultPolicy) {
  const state = previous || newMouthTrack();
  // Accepts one face's landmarks or a list of faces (MediaPipe faceLandmarks).
  const list = !landmarksList ? [] : Array.isArray(landmarksList[0]) ? landmarksList
    : landmarksList.length ? [landmarksList] : [];
  const faces = list.map((landmarks) => mouthGeometry(landmarks, width, height)).filter(Boolean);

  if (state.face) {
    const match = nearestConsistent(faces, state.face, width, height, policy);
    if (match) {
      const candidate = cropROI(match, state.scaleRatio, width, height);
      const elapsed = clamp(nowMs - state.lastSeenMs, 1, 160);
      const a = 1 - Math.exp(-elapsed / 65), b = 1 - Math.exp(-elapsed / 220);
      const roi = clampROI({ cx: state.roi.cx + a * (candidate.cx - state.roi.cx),
        cy: state.roi.cy + a * (candidate.cy - state.roi.cy),
        size: state.roi.size + b * (candidate.size - state.roi.size) }, width, height);
      return { state: lockOn(match, roi, nowMs, state.scaleRatio), status: 'tracking', roi, lips: match.lips, coasted: false };
    }
    if (nowMs - state.lastSeenMs <= policy.graceMs) {
      // Coast on the last ROI; public status and inference window are untouched.
      return { state, status: 'tracking', roi: state.roi, lips: [], coasted: true };
    }
    const next = { ...newMouthTrack(), lostSinceMs: state.lastSeenMs, priorTarget: state.face, holdRoi: state.roi };
    return reacquire(next, faces, width, height, nowMs, policy);
  }
  return reacquire(state, faces, width, height, nowMs, policy);
}

function reacquire(state, faces, width, height, nowMs, policy) {
  const lostSinceMs = state.lostSinceMs ?? nowMs;
  state = { ...state, lostSinceMs };
  if (!faces.length) return lost(state, { candidateFace: null, candidateStreak: 0, candidateStableMs: null });
  const target = state.priorTarget;
  const match = target ? nearestConsistent(faces, target, width, height, policy) : null;
  if (match) return relock(state, match, true, policy, nowMs, width, height);
  if (faces.length > 1) return lost(state, { candidateFace: null, candidateStreak: 0, candidateStableMs: null,
    message: '检测到多个人脸，无法确认目标，请点击「重新捕捉」。' });
  // A single, inconsistent face (or no prior target to compare against): require a
  // longer stable period before trusting it automatically.
  if (!target) return relock(state, faces[0], false, policy, nowMs, width, height);
  return relockSlow(state, faces[0], policy, nowMs, width, height);
}
function sameCandidate(state, face, width, height, policy) {
  return state.candidateFace ? consistent(face.face, state.candidateFace, width, height, policy) : false;
}
function relock(state, candidate, requireStreak, policy, nowMs, width, height) {
  const streak = sameCandidate(state, candidate, width, height, policy) ? state.candidateStreak + 1 : 1;
  if (!requireStreak || streak >= policy.relockFrames) {
    const roi = cropROI(candidate, candidate.scaleRatio, width, height);
    return { state: lockOn(candidate, roi, nowMs, candidate.scaleRatio), status: 'tracking', roi,
      lips: candidate.lips, coasted: false, relocked: requireStreak ? 'consistent' : 'first' };
  }
  return lost(state, { candidateFace: candidate.face, candidateStreak: streak, candidateStableMs: null,
    message: '检测到可能的目标，正在确认…' });
}
function relockSlow(state, candidate, policy, nowMs, width, height) {
  const matchesPrior = sameCandidate(state, candidate, width, height, policy);
  const candidateStableMs = matchesPrior ? state.candidateStableMs ?? nowMs : nowMs;
  const streak = matchesPrior ? state.candidateStreak + 1 : 1;
  if (nowMs - candidateStableMs >= policy.inconsistentRelockMs) {
    const roi = cropROI(candidate, candidate.scaleRatio, width, height);
    return { state: lockOn(candidate, roi, nowMs, candidate.scaleRatio), status: 'tracking', roi,
      lips: candidate.lips, coasted: false, relocked: 'inconsistent' };
  }
  return lost(state, { candidateFace: candidate.face, candidateStreak: streak, candidateStableMs,
    message: '目标位置变化较大，稳定后将自动确认，或点击「重新捕捉」。' });
}
