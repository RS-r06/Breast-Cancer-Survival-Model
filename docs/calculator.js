// Reads the form, runs the model (model.js) and draws the survival curve.

const form = document.getElementById("inputs");
const canvas = document.getElementById("curve");
let model = null;

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function readProfile() {
  const f = new FormData(form);
  const num = (k) => Number(f.get(k));
  return {
    age: num("age"), meno: num("meno"), size_mm: num("size_mm"), grade: num("grade"),
    nodes: Math.max(1, num("nodes")), pgr: num("pgr"), er: num("er"), hormon: num("hormon"),
  };
}

function averageCurve() {
  // The baseline survival is the curve at the development cohort's mean profile.
  return model.baseline_survival;
}

function draw(curve) {
  const ratio = window.devicePixelRatio || 1;
  const { width, height } = canvas.getBoundingClientRect();
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  const pad = { l: 44, r: 12, t: 10, b: 30 };
  const x = (t) => pad.l + (t / 5) * (width - pad.l - pad.r);
  const y = (s) => pad.t + (1 - s) * (height - pad.t - pad.b);

  ctx.fillStyle = css("--panel");
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = css("--line");
  ctx.fillStyle = css("--muted");
  ctx.font = "12px system-ui, sans-serif";
  ctx.lineWidth = 1;
  for (let s = 0; s <= 1.0001; s += 0.25) {
    ctx.beginPath();
    ctx.moveTo(pad.l, y(s));
    ctx.lineTo(width - pad.r, y(s));
    ctx.stroke();
    ctx.fillText(`${Math.round(s * 100)}%`, 4, y(s) + 4);
  }
  for (let t = 0; t <= 5; t++) ctx.fillText(`${t}y`, x(t) - 6, height - 10);

  const line = (values, color, width_, dash) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width_;
    ctx.setLineDash(dash);
    ctx.beginPath();
    values.forEach((s, i) => {
      const px = x(model.times[i]);
      if (i === 0) ctx.moveTo(px, y(s));
      else ctx.lineTo(px, y(s));
    });
    ctx.stroke();
    ctx.setLineDash([]);
  };
  line(averageCurve(), css("--muted"), 1.5, [5, 4]);
  line(curve, css("--accent"), 2.5, []);
}

function update() {
  if (!model || !form.checkValidity()) return;
  const curve = survivalCurve(model, readProfile());
  document.getElementById("rfs").textContent = `${Math.round(curve[curve.length - 1] * 100)}%`;
  draw(curve);
}

form.addEventListener("input", update);
window.addEventListener("resize", update);

fetch("model.json")
  .then((r) => r.json())
  .then((m) => {
    model = m;
    const v = m.validation;
    document.getElementById("ndev").textContent = m.development_patients.toLocaleString();
    document.getElementById("validation").textContent =
      `On the German trial it never saw, in ${Math.round(v.c_index * 100)}% of comparable pairs of women the one who relapsed ` +
      `first had the higher predicted risk (concordance ${v.c_index.toFixed(2)}; 50% is a coin toss). Its predicted 5-year ` +
      `risks also matched what happened: observed over expected ` +
      `${v.o_e.toFixed(2)}, calibration slope ${v.slope.toFixed(2)} (1 is perfect for both).`;
    update();
  });
