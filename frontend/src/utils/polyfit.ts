/**
 * Least-squares polynomial fitting in the browser.
 * Replicates numpy.polyfit(x, y, deg) — returns coefficients highest-degree first.
 */

function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length, n = B[0].length, p = B.length;
  return Array.from({length: m}, (_, i) =>
    Array.from({length: n}, (_, j) =>
      Array.from({length: p}, (_, k) => A[i][k] * B[k][j]).reduce((a, b) => a + b, 0)
    )
  );
}

function matVec(A: number[][], v: number[]): number[] {
  return A.map(row => row.reduce((s, a, j) => s + a * v[j], 0));
}

function luSolve(A: number[][], b: number[]): number[] {
  const n = A.length;
  const L: number[][] = Array.from({length: n}, (_, i) => Array.from({length: n}, (_, j) => (i === j ? 1 : 0) as number));
  const U: number[][] = A.map(row => [...row]);
  const perm = Array.from({length: n}, (_, i) => i);

  for (let k = 0; k < n; k++) {
    let maxVal = Math.abs(U[k][k]), maxRow = k;
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(U[i][k]) > maxVal) { maxVal = Math.abs(U[i][k]); maxRow = i; }
    }
    if (maxRow !== k) {
      [U[k], U[maxRow]] = [U[maxRow], U[k]];
      [perm[k], perm[maxRow]] = [perm[maxRow], perm[k]];
      for (let j = 0; j < k; j++) [L[k][j], L[maxRow][j]] = [L[maxRow][j], L[k][j]];
    }
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(U[k][k]) < 1e-14) continue;
      L[i][k] = U[i][k] / U[k][k];
      for (let j = k; j < n; j++) U[i][j] -= L[i][k] * U[k][j];
    }
  }

  const pb = perm.map(i => b[i]);
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    y[i] = pb[i] - L[i].slice(0, i).reduce((s, v, j) => s + v * y[j], 0);
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    if (Math.abs(U[i][i]) < 1e-14) { x[i] = 0; continue; }
    x[i] = (y[i] - U[i].slice(i + 1).reduce((s, v, j) => s + v * x[i + 1 + j], 0)) / U[i][i];
  }
  return x;
}

/** Polynomial fit — returns coefficients [c_deg, c_{deg-1}, ..., c_0] (highest degree first). */
export function polyfit(x: number[], y: number[], deg: number): number[] {
  const n = x.length;
  // Vandermonde matrix columns in DESCENDING degree order (matches numpy)
  const V: number[][] = x.map(xi =>
    Array.from({length: deg + 1}, (_, k) => Math.pow(xi, deg - k))
  );
  const Vt = V[0].map((_, j) => V.map(row => row[j]));
  const VtV = matMul(Vt, V);
  const Vtb = matVec(Vt, y);
  return luSolve(VtV, Vtb);
}

/** Evaluate polynomial with coefficients [c_deg, ..., c_0] at x. */
export function polyval(coeffs: number[], x: number): number {
  return coeffs.reduce((acc, c) => acc * x + c, 0);
}

/**
 * Q-H curve: fit degree-2, return {A, B, C} matching solver formula: head = A*Q² + B*Q + C
 * Solver (pipeline_model.py line 3173): head_curve = A * Q_equiv**2 + B * Q_equiv + C
 * polyfit returns highest-degree first → [Q²-coeff, Q¹-coeff, constant] = [A, B, C]
 */
export function computePumpCoeffs(qhTable: {flow: number; head: number}[]): {A: number; B: number; C: number} {
  if (qhTable.length < 3) return {A: 0, B: 0, C: 100};
  const flows = qhTable.map(r => r.flow);
  const heads = qhTable.map(r => r.head);
  const [A, B, C] = polyfit(flows, heads, 2); // A=Q²-coeff, B=Q¹-coeff, C=constant
  return {A, B, C};
}

/**
 * Efficiency curve: fit degree-4, return {P,Q,R,S,T} matching solver formula:
 * eff = P*Q⁴ + Q*Q³ + R*Q² + S*Q + T
 * Solver (pipeline_model.py line 3195): eff_curve = P*Q**4 + Qc*Q**3 + R*Q**2 + S*Q + T
 * polyfit returns highest-degree first → [Q⁴, Q³, Q², Q¹, constant] = [P, Q, R, S, T]
 */
export function computeEffCoeffs(effTable: {flow: number; eff: number}[]): {P: number; Q: number; R: number; S: number; T: number} {
  if (effTable.length < 5) return {P: 0, Q: 0, R: 0, S: 0, T: 70};
  const flows = effTable.map(r => r.flow);
  const effs = effTable.map(r => r.eff);
  const [P, Q, R, S, T] = polyfit(flows, effs, 4); // P=Q⁴, Q=Q³, R=Q², S=Q¹, T=constant
  return {P, Q, R, S, T};
}
