"use strict";
// A 3D brain made of the model's neurons, drawn with three.js.
//
// Every point is one real neuron. Its depth shows its layer, and its region shows
// its head: left or right, front or back. The shape itself is an illustration, as
// the model's neurons have no anatomy. Nothing lights up unless the server reports
// that it fired, so the brain stays still between messages.

const brain = (() => {
  const LAYERS = 4;
  const HEADS = 4;
  const PER_HEAD = 1024;
  const COUNT = LAYERS * HEADS * PER_HEAD;
  const PATHS = 240; // Line segments kept for the signal moving between layers.

  const canvas = document.getElementById("brain");
  const wrap = document.getElementById("brain-wrap");
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  function randomFor(seed) {
    let state = (seed * 2654435761 + 1013904223) >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(random) {
    return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
  }

  // The ridges of the cortex: high on a fold, low in the groove between folds.
  function folds(x, y, z) {
    return Math.sin(z * 11 + 2.6 * Math.sin(y * 7.5) + x * 3)
      * Math.cos(y * 10 + 2.2 * Math.sin(z * 6.5))
      + 0.45 * Math.sin((x + z) * 19 + y * 5);
  }

  function neuronPosition(index) {
    const random = randomFor(index + 1);
    const layer = Math.floor(index / (HEADS * PER_HEAD));
    const head = Math.floor(index / PER_HEAD) % HEADS;
    const side = head < 2 ? -1 : 1;
    const end = head % 2 === 0 ? 1 : -1;
    const kind = random();

    if (kind < 0.012) { // Brain stem
      const t = random();
      const angle = random() * Math.PI * 2;
      const radius = 0.09 * Math.sqrt(random()) * (1 - t * 0.25);
      return [Math.cos(angle) * radius, -0.42 - t * 0.56, -0.26 - t * 0.14 + Math.sin(angle) * radius];
    }
    if (kind < 0.085) { // Cerebellum, with fine horizontal folds
      const a = gaussian(random);
      const b = gaussian(random);
      const c = gaussian(random);
      const length = Math.hypot(a, b, c) || 1;
      const fold = 1 + 0.05 * Math.sin((b / length) * 34);
      const depth = fold * (1 - layer * 0.12);
      const x = side * Math.abs(a / length) * 0.42 * depth + side * 0.02;
      return [x, -0.56 + (b / length) * 0.25 * depth, -0.7 + (c / length) * 0.3 * depth];
    }

    // Cortex. Favour points on the folds, so the grooves show as dark lines.
    let dx = 0;
    let dy = 0;
    let dz = 0;
    let ridge = 0;
    for (let attempt = 0; attempt < 6; attempt += 1) {
      const a = gaussian(random);
      const b = gaussian(random);
      const c = gaussian(random);
      const length = Math.hypot(a, b, c) || 1;
      dx = Math.abs(a / length) * side;
      dy = b / length;
      dz = Math.abs(c / length) * end;
      ridge = folds(dx, dy, dz);
      // The deep groove along the side of a real brain.
      const groove = Math.abs(dy - (-0.1 + 0.28 * dz));
      const inGroove = groove < 0.045 && Math.abs(dx) > 0.35 && dz > -0.35;
      if (!inGroove && random() < 0.08 + 0.42 * Math.max(0, ridge + 0.55)) break;
    }
    const depth = (1 - layer * 0.14) * (1 + 0.075 * ridge * (layer === 0 ? 1 : 0.4)) + (random() - 0.5) * 0.012;
    let y = dy * 0.76 * depth;
    if (y < 0) y *= 0.68; // A flatter underside.
    y *= 1 - 0.16 * Math.max(0, dz); // Lower at the front.
    // The temporal lobe hangs below the middle of each side.
    const lobe = Math.exp(-(((dz - 0.2) / 0.42) ** 2)) * Math.max(0, -dy) * Math.min(1, Math.abs(dx) * 1.6);
    y -= 0.34 * lobe * depth;
    // Each half curves away from the middle, leaving a clear gap.
    const gap = side * 0.06 * (0.6 + 0.4 * Math.max(0, dy));
    return [dx * 0.66 * depth + gap, y + 0.12, dz * depth];
  }

  // Opaque, so the additive glow fades into the page instead of leaving dark marks.
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setClearColor(0x07080b, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 30);
  const group = new THREE.Group();
  scene.add(group);

  const positions = new Float32Array(COUNT * 3);
  const layers = new Float32Array(COUNT);
  for (let index = 0; index < COUNT; index += 1) {
    positions.set(neuronPosition(index), index * 3);
    layers[index] = Math.floor(index / (HEADS * PER_HEAD));
  }

  const fire = new Float32Array(COUNT); // Fired just now. Decays quickly.
  const recall = new Float32Array(COUNT); // Recalled private data just now.
  const heldNow = new Float32Array(COUNT); // Share of the neuron's memory that is flagged.
  const heldGoal = new Float32Array(COUNT);
  const storedNow = new Float32Array(COUNT); // How much the neuron holds at all.
  const storedGoal = new Float32Array(COUNT);
  const fireOut = new Float32Array(COUNT);
  const redOut = new Float32Array(COUNT);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aLayer", new THREE.BufferAttribute(layers, 1));
  const fireAttribute = new THREE.BufferAttribute(fireOut, 1);
  const redAttribute = new THREE.BufferAttribute(redOut, 1);
  const storedAttribute = new THREE.BufferAttribute(storedNow, 1);
  [fireAttribute, redAttribute, storedAttribute].forEach((a) => a.setUsage(THREE.DynamicDrawUsage));
  geometry.setAttribute("aFire", fireAttribute);
  geometry.setAttribute("aRed", redAttribute);
  geometry.setAttribute("aStored", storedAttribute);

  const pointMaterial = new THREE.ShaderMaterial({
    transparent: true,
    depthTest: false,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { uPixels: { value: 1 }, uCamera: { value: 3 } },
    vertexShader: `
      attribute float aFire;
      attribute float aRed;
      attribute float aStored;
      attribute float aLayer;
      uniform float uPixels;
      uniform float uCamera;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vec4 view = modelViewMatrix * vec4(position, 1.0);
        // Nearer neurons are brighter, which gives the brain its depth.
        float near = clamp(1.0 - (-view.z - (uCamera - 1.0)) * 0.42, 0.08, 1.0);
        float size = 3.4 + aStored * 2.0 + aFire * 8.0 + aRed * 7.0;
        gl_PointSize = size * uPixels * (uCamera / -view.z);
        vec3 rest = vec3(0.42, 0.55, 0.85);
        vec3 stored = vec3(0.72, 0.8, 0.95);
        vec3 lit = vec3(0.56, 0.85, 1.0);
        vec3 hot = vec3(1.0, 0.27, 0.2);
        vec3 color = mix(rest, stored, aStored);
        color = mix(color, lit, clamp(aFire * 1.4, 0.0, 1.0));
        color = mix(color, hot, clamp(aRed * 1.5, 0.0, 1.0));
        vColor = color;
        float base = (0.95 - aLayer * 0.25) + aStored * 0.4;
        vAlpha = clamp(base * near + aFire + aRed, 0.0, 1.0);
        gl_Position = projectionMatrix * view;
      }`,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        float d = length(gl_PointCoord - 0.5) * 2.0;
        if (d > 1.0) discard;
        float glow = smoothstep(1.0, 0.0, d);
        gl_FragColor = vec4(vColor, vAlpha * glow);
      }`,
  });
  group.add(new THREE.Points(geometry, pointMaterial));

  // Lines show the signal passing from one layer to the next for each byte.
  // They are not individual synapses.
  const pathPositions = new Float32Array(PATHS * 6);
  const pathColours = new Float32Array(PATHS * 6);
  const pathLife = new Float32Array(PATHS);
  const pathHot = new Uint8Array(PATHS);
  let nextPath = 0;
  const pathGeometry = new THREE.BufferGeometry();
  pathGeometry.setAttribute("position", new THREE.BufferAttribute(pathPositions, 3).setUsage(THREE.DynamicDrawUsage));
  pathGeometry.setAttribute("color", new THREE.BufferAttribute(pathColours, 3).setUsage(THREE.DynamicDrawUsage));
  group.add(new THREE.LineSegments(pathGeometry, new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthTest: false,
    depthWrite: false,
  })));

  function addPath(from, to, hot) {
    const slot = nextPath;
    nextPath = (nextPath + 1) % PATHS;
    for (let axis = 0; axis < 3; axis += 1) {
      pathPositions[slot * 6 + axis] = positions[from * 3 + axis];
      pathPositions[slot * 6 + 3 + axis] = positions[to * 3 + axis];
    }
    pathLife[slot] = 1;
    pathHot[slot] = hot ? 1 : 0;
  }

  function resize() {
    const box = wrap.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(ratio);
    renderer.setSize(box.width, box.height, false);
    camera.aspect = box.width / Math.max(1, box.height);
    // Keep the whole brain in view when the space is narrow.
    camera.position.z = 3.7 / Math.min(1, camera.aspect * 0.95);
    camera.updateProjectionMatrix();
    pointMaterial.uniforms.uCamera.value = camera.position.z;
    pointMaterial.uniforms.uPixels.value = ratio * Math.max(0.8, Math.min(box.width, box.height) / 520);
  }

  // Turning it

  let yaw = 1.15;
  let pitch = 0.14;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  wrap.addEventListener("pointerdown", (event) => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    wrap.setPointerCapture(event.pointerId);
  });
  wrap.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    yaw += (event.clientX - lastX) * 0.008;
    pitch = Math.max(-1, Math.min(1, pitch + (event.clientY - lastY) * 0.006));
    lastX = event.clientX;
    lastY = event.clientY;
  });
  wrap.addEventListener("pointerup", () => { dragging = false; });
  wrap.addEventListener("pointercancel", () => { dragging = false; });

  let previous = performance.now();
  function animate(now) {
    const dt = Math.min(0.1, (now - previous) / 1000);
    previous = now;
    if (!dragging && !reduceMotion) yaw += dt * 0.07;
    group.rotation.set(pitch, yaw, 0);

    const fireDecay = Math.exp(-dt / 0.2);
    const recallDecay = Math.exp(-dt / 0.45);
    const settle = 1 - Math.exp(-dt / 0.45);
    for (let index = 0; index < COUNT; index += 1) {
      fire[index] *= fireDecay;
      recall[index] *= recallDecay;
      heldNow[index] += (heldGoal[index] - heldNow[index]) * settle;
      storedNow[index] += (storedGoal[index] - storedNow[index]) * settle;
      fireOut[index] = fire[index];
      redOut[index] = Math.max(heldNow[index] * 0.75, recall[index]);
    }
    fireAttribute.needsUpdate = true;
    redAttribute.needsUpdate = true;
    storedAttribute.needsUpdate = true;

    const pathDecay = Math.exp(-dt / 0.28);
    for (let slot = 0; slot < PATHS; slot += 1) {
      pathLife[slot] *= pathDecay;
      const life = pathLife[slot] * 0.7;
      const r = pathHot[slot] ? 1.0 : 0.4;
      const g = pathHot[slot] ? 0.25 : 0.75;
      const b = pathHot[slot] ? 0.2 : 1.0;
      for (let end = 0; end < 2; end += 1) {
        pathColours[slot * 6 + end * 3] = r * life;
        pathColours[slot * 6 + end * 3 + 1] = g * life;
        pathColours[slot * 6 + end * 3 + 2] = b * life;
      }
    }
    pathGeometry.attributes.position.needsUpdate = true;
    pathGeometry.attributes.color.needsUpdate = true;

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  new ResizeObserver(resize).observe(wrap);
  resize();
  requestAnimationFrame(animate);

  // A real signal jumps to a neuron nearby, so link each one to its nearest
  // neighbour among those firing in the next layer, and skip long jumps.
  function linkLayers(layer, next, count, hot) {
    for (let slot = 0; slot < Math.min(count, layer.i.length); slot += 1) {
      const from = layer.i[slot];
      let best = -1;
      let bestDistance = 0.6;
      next.i.forEach((to) => {
        const distance = Math.hypot(
          positions[from * 3] - positions[to * 3],
          positions[from * 3 + 1] - positions[to * 3 + 1],
          positions[from * 3 + 2] - positions[to * 3 + 2],
        );
        if (distance < bestDistance) {
          bestDistance = distance;
          best = to;
        }
      });
      if (best >= 0) addPath(from, best, hot);
    }
  }

  const flatIndex = (n) => n.layer * HEADS * PER_HEAD + n.head * PER_HEAD + n.neuron;

  return {
    // Show what memory holds, from the state's strongest neurons.
    showMemory(neurons) {
      heldGoal.fill(0);
      storedGoal.fill(0);
      let peak = 1e-6;
      neurons.forEach((n) => { peak = Math.max(peak, n.strength); });
      neurons.forEach((n) => {
        const index = flatIndex(n);
        heldGoal[index] = n.flagged_share;
        storedGoal[index] = Math.sqrt(n.strength / peak);
      });
    },

    // Light up one byte's real activity, layer by layer, over ``duration`` ms.
    fireFrame(frame, duration) {
      frame.fired.forEach((layer, level) => {
        setTimeout(() => {
          layer.i.forEach((index, slot) => {
            fire[index] = Math.max(fire[index], layer.s[slot]);
          });
          if (level + 1 < frame.fired.length) linkLayers(layer, frame.fired[level + 1], 6, false);
        }, (level / frame.fired.length) * duration);
      });
      if (frame.depends > 0.05) {
        frame.flagged.forEach((layer, level) => {
          layer.i.forEach((index, slot) => {
            recall[index] = Math.max(recall[index], layer.s[slot] * frame.depends);
          });
          const next = frame.flagged[level + 1];
          if (next && frame.depends > 0.5) linkLayers(layer, next, 6, true);
        });
      }
    },

    // The neurons holding the forgotten value flare as they are erased.
    flareHeld() {
      for (let index = 0; index < COUNT; index += 1) {
        if (heldNow[index] > 0.1) recall[index] = 1;
      }
    },
  };
})();
