"use strict";
// A glass brain holding the model's real neurons, drawn with three.js.
//
// Every dot is one neuron. The four regions are the model's four heads. Within a
// region, a neuron's place comes from its learned weights, so neighbours respond
// to similar characters. The outline is an illustration: the model has no anatomy.
// Nothing lights up unless the server reports it, and the shell's glow follows the
// reported activity, so the brain rests between messages.

const brain = (() => {
  const PATHS = 60;
  const canvas = document.getElementById("brain");
  const wrap = document.getElementById("brain-wrap");

  // The outline. A direction from the centre gives a point on the folded surface.

  function folds(x, y, z) {
    return Math.sin(z * 12 + 2.8 * Math.sin(y * 8) + x * 3)
      * Math.cos(y * 11 + 2.4 * Math.sin(z * 7))
      + 0.4 * Math.sin((x + z) * 21 + y * 6);
  }

  function cortex(dx, dy, dz, side) {
    const ridge = folds(dx, dy, dz);
    // The deep groove along the side of a real brain.
    const groove = Math.abs(dy - (-0.1 + 0.28 * dz));
    const dip = Math.abs(dx) > 0.3 && dz > -0.4 ? Math.max(0, 1 - groove / 0.05) * 0.05 : 0;
    const depth = 1 + 0.06 * ridge - dip;
    let y = dy * 0.76 * depth;
    if (y < 0) y *= 0.68; // A flatter underside.
    y *= 1 - 0.16 * Math.max(0, dz); // Lower at the front.
    // The temporal lobe hangs below the middle of each side.
    const lobe = Math.exp(-(((dz - 0.2) / 0.42) ** 2)) * Math.max(0, -dy) * Math.min(1, Math.abs(dx) * 1.6);
    y -= 0.34 * lobe * depth;
    const gap = side * 0.055 * (0.6 + 0.4 * Math.max(0, dy));
    return [dx * 0.66 * depth + gap, y + 0.1, dz * depth];
  }

  // Where one neuron sits: its head picks the region, and its u and v pick the place.
  function neuronPosition(head, u, v) {
    const side = head < 2 ? -1 : 1;
    const end = head % 2 === 0 ? 1 : -1;
    const y = (v * 2 - 1) * 0.97;
    const ring = Math.sqrt(1 - y * y);
    const angle = u * Math.PI / 2;
    return cortex(Math.cos(angle) * ring * side, y, Math.sin(angle) * ring * end, side);
  }

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setClearColor(0x05060a, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 30);
  const group = new THREE.Group();
  scene.add(group);

  // The glass shell

  const shellUniforms = { uPulse: { value: 0 }, uHeat: { value: 0 }, uFlash: { value: 0 } };
  const shellMaterial = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    uniforms: shellUniforms,
    vertexShader: `
      attribute float aRidge;
      varying vec3 vNormal;
      varying vec3 vView;
      varying float vRidge;
      void main() {
        vec4 view = modelViewMatrix * vec4(position, 1.0);
        vRidge = aRidge;
        vNormal = normalize(normalMatrix * normal);
        vView = normalize(-view.xyz);
        gl_Position = projectionMatrix * view;
      }`,
    fragmentShader: `
      uniform float uPulse;
      uniform float uHeat;
      uniform float uFlash;
      varying vec3 vNormal;
      varying vec3 vView;
      varying float vRidge;
      void main() {
        vec3 normal = normalize(vNormal);
        float facing = abs(dot(normal, vView));
        float rim = pow(1.0 - facing, 2.2);
        float light = max(0.0, dot(normal, normalize(vec3(0.4, 0.8, 0.5))));
        vec3 cool = vec3(0.16, 0.34, 0.62);
        vec3 hot = vec3(1.0, 0.22, 0.14);
        vec3 tint = mix(cool, hot, uHeat);
        // Folds catch the light and grooves stay dark, as on a real cortex.
        float fold = smoothstep(-0.35, 0.75, vRidge);
        float strength = fold * (0.05 + light * 0.1 + uPulse * 0.03 + uHeat * 0.06)
          + rim * (0.14 + uPulse * 0.12 + uHeat * 0.22);
        gl_FragColor = vec4(tint * strength + vec3(uFlash) * rim * 0.5, 1.0);
      }`,
  });

  [-1, 1].forEach((side) => {
    const shell = new THREE.SphereGeometry(1, 260, 200, side < 0 ? -Math.PI / 2 : Math.PI / 2, Math.PI);
    const vertices = shell.attributes.position;
    const ridges = new Float32Array(vertices.count);
    for (let index = 0; index < vertices.count; index += 1) {
      const x = vertices.getX(index);
      const y = vertices.getY(index);
      const z = vertices.getZ(index);
      ridges[index] = folds(x, y, z);
      const point = cortex(x, y, z, side);
      vertices.setXYZ(index, point[0], point[1], point[2]);
    }
    shell.setAttribute("aRidge", new THREE.BufferAttribute(ridges, 1));
    shell.computeVertexNormals();
    group.add(new THREE.Mesh(shell, shellMaterial));
  });

  // The neurons, created when the server describes the model

  let count = 0;
  let positions = null;
  let fire = null; // Fired just now.
  let recall = null; // Recalled private data just now.
  let heldNow = null; // Share of the neuron's memory that is flagged.
  let heldGoal = null;
  let storedNow = null; // How much the neuron holds at all.
  let storedGoal = null;
  let fireOut = null;
  let redOut = null;
  let fireAttribute = null;
  let redAttribute = null;
  let storedAttribute = null;

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
      uniform float uPixels;
      uniform float uCamera;
      varying vec3 vColor;
      varying float vAlpha;
      varying float vLit;
      void main() {
        vec4 view = modelViewMatrix * vec4(position, 1.0);
        float near = clamp(1.05 - (-view.z - (uCamera - 1.0)) * 0.38, 0.3, 1.0);
        float lit = max(aFire, aRed);
        float size = 4.2 + aStored * 3.0 + lit * 22.0;
        gl_PointSize = size * uPixels * (uCamera / -view.z);
        vec3 rest = vec3(0.34, 0.5, 0.86);
        vec3 stored = vec3(0.7, 0.82, 1.0);
        vec3 firing = vec3(0.45, 0.88, 1.0);
        vec3 hot = vec3(1.0, 0.2, 0.13);
        vec3 color = mix(rest, stored, aStored);
        color = mix(color, firing, clamp(aFire * 2.0, 0.0, 1.0));
        color = mix(color, hot, clamp(aRed * 1.8, 0.0, 1.0));
        vColor = color;
        vLit = lit;
        vAlpha = clamp((0.42 + aStored * 0.45) * near + lit * 1.0, 0.0, 1.0);
        gl_Position = projectionMatrix * view;
      }`,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;
      varying float vLit;
      void main() {
        float d = length(gl_PointCoord - 0.5) * 2.0;
        if (d > 1.0) discard;
        // A bright core inside a soft halo, so firing neurons look like they glow.
        float halo = pow(1.0 - d, 2.0);
        float core = smoothstep(0.3, 0.06, d) * (0.4 + vLit * 0.6);
        gl_FragColor = vec4(vColor * (1.0 + core * 0.7) + vec3(core * vLit * 0.4), vAlpha * (halo * 0.7 + core));
      }`,
  });

  // Lines are learned connections between neurons firing in consecutive rounds.
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
    camera.position.z = 3.2 / Math.min(1, camera.aspect * 0.95);
    camera.updateProjectionMatrix();
    pointMaterial.uniforms.uCamera.value = camera.position.z;
    pointMaterial.uniforms.uPixels.value = ratio * Math.max(0.8, Math.min(box.width, box.height) / 560);
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

  let pulseGoal = 0;
  let heatGoal = 0;
  let previous = performance.now();

  function animate(now) {
    const dt = Math.min(0.1, (now - previous) / 1000);
    previous = now;
    if (!dragging) yaw += dt * 0.09;
    group.rotation.set(pitch, yaw, 0);

    // Quick enough that each character's neurons can be told from the next ones.
    const fireDecay = Math.exp(-dt / 0.2);
    const recallDecay = Math.exp(-dt / 0.6);
    const settle = 1 - Math.exp(-dt / 0.5);
    for (let index = 0; index < count; index += 1) {
      fire[index] *= fireDecay;
      recall[index] *= recallDecay;
      heldNow[index] += (heldGoal[index] - heldNow[index]) * settle;
      storedNow[index] += (storedGoal[index] - storedNow[index]) * settle;
      fireOut[index] = fire[index];
      // Neurons with little flagged data stay unlit, so the red marks clear holders.
      const held = Math.max(0, Math.min(1, (heldNow[index] - 0.25) / 0.6));
      redOut[index] = Math.max(held * 0.5, recall[index]);
    }
    if (count) {
      fireAttribute.needsUpdate = true;
      redAttribute.needsUpdate = true;
      storedAttribute.needsUpdate = true;
    }

    pulseGoal *= Math.exp(-dt / 0.4);
    heatGoal *= Math.exp(-dt / 0.7);
    shellUniforms.uPulse.value += (pulseGoal - shellUniforms.uPulse.value) * (1 - Math.exp(-dt / 0.08));
    shellUniforms.uHeat.value += (heatGoal - shellUniforms.uHeat.value) * (1 - Math.exp(-dt / 0.12));
    shellUniforms.uFlash.value *= Math.exp(-dt / 0.35);

    const pathDecay = Math.exp(-dt / 0.28);
    for (let slot = 0; slot < PATHS; slot += 1) {
      pathLife[slot] *= pathDecay;
      const life = pathLife[slot] * 0.22;
      const r = pathHot[slot] ? 1.0 : 0.45;
      const g = pathHot[slot] ? 0.28 : 0.85;
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

  return {
    // Build one dot per real neuron from the server's description of the model.
    build(model) {
      if (count) return;
      count = model.neurons;
      positions = new Float32Array(count * 3);
      for (let index = 0; index < count; index += 1) {
        const head = Math.floor(index / model.per_head);
        positions.set(neuronPosition(head, model.u[index], model.v[index]), index * 3);
      }
      fire = new Float32Array(count);
      recall = new Float32Array(count);
      heldNow = new Float32Array(count);
      heldGoal = new Float32Array(count);
      storedNow = new Float32Array(count);
      storedGoal = new Float32Array(count);
      fireOut = new Float32Array(count);
      redOut = new Float32Array(count);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      fireAttribute = new THREE.BufferAttribute(fireOut, 1).setUsage(THREE.DynamicDrawUsage);
      redAttribute = new THREE.BufferAttribute(redOut, 1).setUsage(THREE.DynamicDrawUsage);
      storedAttribute = new THREE.BufferAttribute(storedNow, 1).setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute("aFire", fireAttribute);
      geometry.setAttribute("aRed", redAttribute);
      geometry.setAttribute("aStored", storedAttribute);
      group.add(new THREE.Points(geometry, pointMaterial));
    },

    // Show what memory holds: per neuron, its strength and its flagged share.
    showMemory(memory) {
      for (let index = 0; index < count; index += 1) {
        storedGoal[index] = memory.strength[index] / 100;
        heldGoal[index] = memory.flagged[index] / 100;
      }
    },

    // Light up one byte's real activity, round by round, over ``duration`` ms.
    fireFrame(frame, duration) {
      frame.fired.forEach((round, level) => {
        setTimeout(() => {
          round.i.forEach((index, slot) => {
            // Every reported neuron did fire, so even the weakest must be seen.
            fire[index] = Math.max(fire[index], 0.5 + 0.5 * round.s[slot]);
          });
          const links = frame.links[level - 1];
          if (links) links.forEach(([from, to]) => addPath(from, to, frame.depends > 0.5));
        }, (level / frame.fired.length) * duration);
      });
      const firing = frame.counts.reduce((sum, value) => sum + value, 0) / frame.counts.length;
      pulseGoal = Math.min(1, (firing / count) * 2.5);
      if (frame.depends > 0.05) {
        heatGoal = Math.max(heatGoal, frame.depends);
        frame.flagged.forEach((round) => {
          round.i.forEach((index, slot) => {
            recall[index] = Math.max(recall[index], round.s[slot] * frame.depends);
          });
        });
      }
    },

    // The neurons holding the forgotten value flare as it is erased.
    flareHeld() {
      for (let index = 0; index < count; index += 1) {
        if (heldNow[index] > 0.25) recall[index] = 1;
      }
      shellUniforms.uFlash.value = 0.5;
      heatGoal = 1;
    },
  };
})();
