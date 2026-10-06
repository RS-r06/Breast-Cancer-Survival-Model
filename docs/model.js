// The model's prediction, in the browser. Must match bcsurv/cox.py exactly;
// tests/test_calculator.py runs this file in Node and compares.

function featureVector(p) {
  return [
    p.age,
    p.meno,
    p.size_mm > 20 && p.size_mm <= 50 ? 1 : 0,
    p.size_mm > 50 ? 1 : 0,
    p.grade === 3 ? 1 : 0,
    Math.log1p(p.nodes),
    Math.log1p(p.pgr),
    Math.log1p(p.er),
    p.hormon,
  ];
}

// S(t | x) = S0(t) ^ exp((x - mean) . beta), at each time on the model's grid.
function survivalCurve(model, p) {
  const x = featureVector(p);
  let lp = 0;
  for (let i = 0; i < x.length; i++) lp += (x[i] - model.means[i]) * model.beta[i];
  const factor = Math.exp(lp);
  return model.baseline_survival.map((s0) => Math.pow(s0, factor));
}

function riskAt5(model, p) {
  const curve = survivalCurve(model, p);
  return 1 - curve[curve.length - 1];
}

if (typeof module !== "undefined") module.exports = { featureVector, survivalCurve, riskAt5 };
