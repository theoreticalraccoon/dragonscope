"use strict";
// A tiny, untrained BDH calculated live in the browser. See bdh.html.
//
// ToyBDH follows BDH.forward in dragonscope/vendor/bdh/bdh.py, using the running
// memory rho from dragonscope/engine.py. Its weights are random. The synapse matrix
// sigma is the paper's Hebbian rule, kept here only so that synapses can be drawn;
// it does not affect the calculation.

const $ = (id) => document.getElementById(id);

class ToyBDH {
  constructor(seed = 20251) {
    this.rounds = 3;
    this.heads = 4;
    this.D = 32;
    this.N = 32; // Per head, so 128 neurons.
    this.neurons = this.heads * this.N;
    const random = this.seeded(seed);
    const normal = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random()) * 0.02;
    const fill = (count) => Float64Array.from({ length: count }, normal);
    this.embed = Array.from({ length: 256 }, () => fill(this.D));
    this.encoder = Array.from({ length: this.heads }, () => fill(this.D * this.N)); // [head][d * N + n]
    this.encoderV = Array.from({ length: this.heads }, () => fill(this.D * this.N));
    this.decoder = fill(this.neurons * this.D); // [neuron * D + d]
    // The same frequencies as get_freqs in the vendored code.
    this.freqs = Float64Array.from({ length: this.N }, (_, n) => (
      1 / (65536 ** ((Math.floor(n / 2) * 2) / this.N)) / (2 * Math.PI)
    ));
    this.reset();
  }

  seeded(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  reset() {
    this.position = 0;
    // rho[round][head] is D * N, the compressed synaptic state.
    this.rho = Array.from({ length: this.rounds }, () => (
      Array.from({ length: this.heads }, () => new Float64Array(this.D * this.N))
    ));
    this.sigma = new Float64Array(this.neurons * this.neurons);
  }

  // Unit mean and variance over the vector, like nn.LayerNorm with no learned scale.
  layerNorm(vector) {
    const mean = vector.reduce((sum, value) => sum + value, 0) / vector.length;
    const variance = vector.reduce((sum, value) => sum + (value - mean) ** 2, 0) / vector.length;
    const scale = Math.sqrt(variance + 1e-5);
    return vector.map((value) => (value - mean) / scale);
  }

  // Attention.rope: rotate pairs of neurons by an angle that depends on position.
  rotate(values, position) {
    const out = new Float64Array(this.N);
    for (let n = 0; n < this.N; n += 1) {
      const phase = ((position * this.freqs[n]) % 1) * 2 * Math.PI;
      const partner = n % 2 === 0 ? -values[n + 1] : values[n - 1];
      out[n] = values[n] * Math.cos(phase) + partner * Math.sin(phase);
    }
    return out;
  }

  // One character, as a series of events that can be shown one at a time.
  * readByte(byte) {
    const { D, N, heads } = this;
    const position = this.position;
    let signal = this.layerNorm(Float64Array.from(this.embed[byte]));
    yield { stage: "read", byte };
    let previous = null;

    for (let round = 0; round < this.rounds; round += 1) {
      // Fire: x_sparse = relu(x @ encoder)
      const x = new Float64Array(this.neurons);
      for (let head = 0; head < heads; head += 1) {
        for (let n = 0; n < N; n += 1) {
          let sum = 0;
          for (let d = 0; d < D; d += 1) sum += signal[d] * this.encoder[head][d * N + n];
          x[head * N + n] = Math.max(0, sum);
        }
      }
      yield { stage: "fire", byte, round, x };

      // Wire together: the paper's rule, using the previous round's passed signal.
      if (previous) {
        let touched = 0;
        for (let i = 0; i < this.neurons; i += 1) {
          if (!previous[i]) continue;
          for (let j = 0; j < this.neurons; j += 1) {
            if (x[j] && i !== j) {
              this.sigma[i * this.neurons + j] += previous[i] * x[j];
              touched += 1;
            }
          }
        }
        yield { stage: "wire", byte, round, x, touched };
      }

      // Recall: query memory, then store this character. y gates the neurons.
      const y = new Float64Array(this.neurons);
      for (let head = 0; head < heads; head += 1) {
        const qr = this.rotate(x.subarray(head * N, (head + 1) * N), position);
        const memory = this.rho[round][head];
        const attention = new Float64Array(D);
        for (let d = 0; d < D; d += 1) {
          for (let n = 0; n < N; n += 1) attention[d] += memory[d * N + n] * qr[n];
        }
        for (let d = 0; d < D; d += 1) {
          for (let n = 0; n < N; n += 1) memory[d * N + n] += signal[d] * qr[n];
        }
        const key = this.layerNorm(attention);
        for (let n = 0; n < N; n += 1) {
          let sum = 0;
          for (let d = 0; d < D; d += 1) sum += key[d] * this.encoderV[head][d * N + n];
          y[head * N + n] = Math.max(0, sum);
        }
      }
      yield { stage: "recall", byte, round, x, y };

      // Pass it on: only neurons both excited and recalled reach the decoder.
      const passed = x.map((value, index) => value * y[index]);
      const output = new Float64Array(D);
      for (let i = 0; i < this.neurons; i += 1) {
        if (!passed[i]) continue;
        for (let d = 0; d < D; d += 1) output[d] += passed[i] * this.decoder[i * D + d];
      }
      const update = this.layerNorm(output);
      signal = this.layerNorm(signal.map((value, d) => value + update[d]));
      previous = passed;
      yield { stage: "pass", byte, round, x, y, passed };
    }
    this.position += 1;
    this.lastSignal = signal;
  }
}

// Drawing: four arcs of neurons, with synapses as curves across the circle.

const FIRE = [143, 216, 255];
const RECALL = [242, 185, 80];
const PASS = [255, 252, 245];
const REST = [70, 82, 110];
const SYNAPSE = [139, 156, 255];
const MAX_SYNAPSES = 240;

const canvas = $("net");
const context = canvas.getContext("2d");
const toy = new ToyBDH();
let width = 0;
let height = 0;
const glowFire = new Float32Array(toy.neurons);
const glowRecall = new Float32Array(toy.neurons);
const glowPass = new Float32Array(toy.neurons);

function neuronPlace(index) {
  const head = Math.floor(index / toy.N);
  const n = index % toy.N;
  const gap = 0.16;
  const span = Math.PI / 2 - gap;
  const angle = head * (Math.PI / 2) + gap / 2 + (n / (toy.N - 1)) * span - Math.PI / 2;
  const radius = Math.min(width * 0.86, height) / 2 - 18;
  return { x: width / 2 + Math.cos(angle) * radius, y: height / 2 + Math.sin(angle) * radius };
}

const mix = (colour, alpha) => `rgba(${colour[0]},${colour[1]},${colour[2]},${alpha})`;

function draw() {
  context.clearRect(0, 0, width, height);
  // The strongest synapses, thicker and brighter as they grow.
  const links = [];
  for (let k = 0; k < toy.sigma.length; k += 1) if (toy.sigma[k] > 0) links.push(k);
  links.sort((a, b) => toy.sigma[b] - toy.sigma[a]);
  const strongest = links.slice(0, MAX_SYNAPSES);
  const peak = strongest.length ? toy.sigma[strongest[0]] : 1;
  context.lineCap = "round";
  for (let order = strongest.length - 1; order >= 0; order -= 1) {
    const key = strongest[order];
    const from = neuronPlace(Math.floor(key / toy.neurons));
    const to = neuronPlace(key % toy.neurons);
    const share = toy.sigma[key] / peak;
    context.strokeStyle = mix(SYNAPSE, 0.05 + share * 0.55);
    context.lineWidth = 0.5 + share * 1.6;
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.quadraticCurveTo(
      width / 2 + (from.x + to.x - width) * 0.14,
      height / 2 + (from.y + to.y - height) * 0.14,
      to.x,
      to.y,
    );
    context.stroke();
  }
  // Neurons, with the strongest current activity on top.
  for (let index = 0; index < toy.neurons; index += 1) {
    const { x, y } = neuronPlace(index);
    const fire = glowFire[index];
    const recall = glowRecall[index];
    const pass = glowPass[index];
    const lit = Math.max(fire, recall, pass);
    const colour = pass >= Math.max(fire, recall) && pass > 0 ? PASS : recall > fire ? RECALL : lit > 0 ? FIRE : REST;
    if (lit > 0.02) {
      const halo = context.createRadialGradient(x, y, 0, x, y, 8 + lit * 22);
      halo.addColorStop(0, mix(colour, 0.5 * lit));
      halo.addColorStop(1, mix(colour, 0));
      context.fillStyle = halo;
      context.beginPath();
      context.arc(x, y, 8 + lit * 22, 0, Math.PI * 2);
      context.fill();
    }
    context.fillStyle = lit > 0.02 ? mix(colour, 0.5 + lit * 0.5) : mix(REST, 1);
    context.beginPath();
    context.arc(x, y, 3 + lit * 5, 0, Math.PI * 2);
    context.fill();
  }
}

function resize() {
  const box = canvas.parentElement.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  width = box.width;
  height = box.height;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
}

// Scale activity for display, in order: the strongest neuron in the event is 1.
function setGlow(target, values) {
  const peak = values.reduce((best, value) => Math.max(best, value), 1e-9);
  values.forEach((value, index) => { target[index] = Math.max(target[index], Math.sqrt(value / peak)); });
}

// Running through the text

let running = null;
let playing = true;
let waited = 0;
let charactersDone = 0;

function* readAll(text) {
  for (const character of text) {
    const byte = Math.min(255, character.charCodeAt(0));
    yield* toy.readByte(byte);
    charactersDone += 1;
  }
}

function restart() {
  toy.reset();
  charactersDone = 0;
  [glowFire, glowRecall, glowPass].forEach((glow) => glow.fill(0));
  running = readAll($("text").value || " ");
  waited = 0;
  playing = true;
  $("r-char").textContent = "–";
  $("r-round").textContent = "–";
  $("r-fire").textContent = "0";
  $("r-links").textContent = "0";
  showStage(null);
  updatePlay();
}

function showStage(stage) {
  [...$("stages").children].forEach((item) => item.classList.toggle("current", item.dataset.stage === stage));
}

function updatePlay() {
  $("play").textContent = playing ? "Pause" : running ? "Play" : "Play again";
}

function advance() {
  if (!running) return false;
  const { value: event, done } = running.next();
  if (done) {
    running = null;
    playing = false;
    showStage(null);
    $("r-round").textContent = "–";
    $("r-char").textContent = "done";
    updatePlay();
    return false;
  }
  showStage(event.stage);
  $("live-note").textContent = charactersDone === 0
    ? "First character: memory is empty, so nothing can be recalled yet."
    : "";
  const character = String.fromCharCode(event.byte);
  $("r-char").textContent = character === " " ? "space" : character;
  if (event.round !== undefined) $("r-round").textContent = String(event.round + 1);
  if (event.x) {
    $("r-fire").textContent = String(event.x.reduce((count, value) => count + (value > 0 ? 1 : 0), 0));
  }
  if (event.stage === "fire") setGlow(glowFire, event.x);
  if (event.stage === "recall") setGlow(glowRecall, event.y);
  if (event.stage === "pass") setGlow(glowPass, event.passed);
  if (event.stage === "wire") {
    let links = 0;
    for (const value of toy.sigma) if (value > 0) links += 1;
    $("r-links").textContent = links.toLocaleString("en-US");
  }
  return true;
}

// Slow enough to follow while memory forms, and then it picks up.
function stepTime() {
  if (charactersDone === 0) return 320;
  return charactersDone < 3 ? 520 : 110;
}

let previous = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - previous) / 1000);
  previous = now;
  if (playing) {
    waited += dt * 1000;
    if (waited >= stepTime()) {
      waited = 0;
      advance();
    }
  }
  const fade = Math.exp(-dt / (charactersDone < 3 ? 0.5 : 0.22));
  for (let index = 0; index < toy.neurons; index += 1) {
    glowFire[index] *= fade;
    glowRecall[index] *= fade;
    glowPass[index] *= fade;
  }
  draw();
  requestAnimationFrame(frame);
}

$("play").addEventListener("click", () => {
  if (!running) { restart(); return; }
  playing = !playing;
  updatePlay();
});
$("step").addEventListener("click", () => {
  if (!running) restart();
  playing = false;
  updatePlay();
  advance();
});
$("restart").addEventListener("click", restart);
$("text").addEventListener("keydown", (event) => { if (event.key === "Enter") restart(); });

const weights = 2 * toy.heads * toy.D * toy.N + toy.neurons * toy.D + 256 * toy.D;
$("toy-weights").textContent = weights.toLocaleString("en-US");
new ResizeObserver(resize).observe(canvas.parentElement);
resize();
restart();
requestAnimationFrame(frame);
