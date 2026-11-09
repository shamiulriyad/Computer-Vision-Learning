/*
 * app.js — fills the page sections with the components from components.js,
 * using the matrices in lab-data.js. Each block below matches a section of
 * convolution_lab.html.
 */
(function () {
  'use strict';

  const M = window.ConvMath;
  const U = window.ConvUI;
  const D = window.LAB_DATA;
  const h = U.h;
  const fmt = M.fmt;
  const $ = (id) => document.getElementById(id);
  const grid = (m, opts) => new U.MatrixGrid(m, opts).root;

  const W = D.worked;
  const main = M.crossCorrelate(W.input, W.kernel);           // the Day-4 example, S = 1, P = 0
  U.selfCheck('Worked example', main.output, W.expected);

  /* ---------------------------------------------------------------- 1. setup */
  $('setup-input').appendChild(grid(W.input, { label: 'Input image I (5×5)', indices: true }));
  $('setup-kernel').appendChild(grid(W.kernel, { label: 'Kernel K (3×3)', indices: true }));
  $('setup-size').innerHTML = U.formulaHTML(5, 3, 0, 1, { what: 'N = 5, F = 3, P = 0, S = 1' });

  /* ---------------------------------------------------------------- 2. worked example */
  const posSelect = $('diagram-pos');
  main.steps.forEach((s, k) => posSelect.appendChild(new Option('Position ' + (k + 1) + ': O[' + s.outRow + '][' + s.outCol + ']', k)));
  const drawDiagram = () => {
    const mount = $('worked-diagram');
    mount.textContent = '';
    mount.appendChild(U.oneValueDiagram(main, +posSelect.value));
  };
  posSelect.addEventListener('change', drawDiagram);
  drawDiagram();

  main.steps.forEach((_, k) => $('worked-steps').appendChild(U.stepCard(main, k, { mini: true })));
  $('worked-final').appendChild(grid(main.output, { label: 'Feature map O (3×3)', size: 'lg', indices: true, className: 'matrix--final' }));

  // Independent check: for the (+1, 0, −1) kernel, output = left-column sum − right-column sum.
  const colSum = (region, c) => region.map((row) => row[c]);
  let rows = '';
  main.steps.forEach((s) => {
    const left = colSum(s.region, 0), right = colSum(s.region, s.w - 1);
    const L = left.reduce((a, b) => a + b, 0), R = right.reduce((a, b) => a + b, 0);
    rows += '<tr><td>O[' + s.outRow + '][' + s.outCol + ']</td>' +
      '<td>' + left.map(fmt).join(' + ') + ' = ' + fmt(L) + '</td>' +
      '<td>' + right.map(fmt).join(' + ') + ' = ' + fmt(R) + '</td>' +
      '<td>' + fmt(L) + ' − ' + M.fmtTerm(R) + ' = <b>' + fmt(L - R) + '</b></td>' +
      '<td>' + (L - R === s.value ? '✓ matches' : '✗ mismatch') + '</td></tr>';
  });
  $('worked-shortcut').innerHTML = '<table class="vars"><tr><th>Output</th><th>Left column sum</th><th>Right column sum</th><th>Left − Right</th><th>Full calculation</th></tr>' + rows + '</table>';

  /* ---------------------------------------------------------------- 3. animator */
  const cfg = { input: $('cfg-input'), kernel: $('cfg-kernel'), stride: $('cfg-stride'), padding: $('cfg-padding') };
  Object.entries(D.inputs).forEach(([key, v]) => cfg.input.appendChild(new Option(v.label, key)));
  Object.entries(D.kernels).forEach(([key, v]) => cfg.kernel.appendChild(new Option(v.label, key)));
  cfg.kernel.appendChild(new Option('Custom (type your own)', 'custom'));

  // Custom kernel editor: nine number boxes
  const editor = $('kernel-editor');
  const editorCells = [];
  for (let i = 0; i < 9; i++) {
    const inp = h('input', { type: 'number', class: 'num', step: 'any', 'aria-label': 'Kernel row ' + Math.floor(i / 3) + ' column ' + (i % 3) });
    inp.addEventListener('input', () => applyConfig());
    editorCells.push(inp);
    editor.appendChild(inp);
  }
  const setEditor = (k) => k.flat().forEach((v, i) => { editorCells[i].value = v; });
  const readEditor = () => {
    const vals = editorCells.map((i) => { const v = parseFloat(i.value); return Number.isFinite(v) ? v : 0; });
    return [vals.slice(0, 3), vals.slice(3, 6), vals.slice(6, 9)];
  };
  setEditor(D.kernels.vertical.matrix);

  const stepper = new U.ConvolutionStepper($('stepper'), {
    input: D.inputs.day4.matrix, kernel: D.kernels.vertical.matrix, stride: 1, padding: 0
  });

  function applyConfig() {
    const custom = cfg.kernel.value === 'custom';
    $('kernel-editor-wrap').hidden = !custom;
    stepper.setConfig({
      input: D.inputs[cfg.input.value].matrix,
      kernel: custom ? readEditor() : D.kernels[cfg.kernel.value].matrix,
      stride: +cfg.stride.value,
      padding: +cfg.padding.value
    });
  }
  cfg.kernel.addEventListener('change', () => {
    if (cfg.kernel.value !== 'custom') setEditor(D.kernels[cfg.kernel.value].matrix);
    applyConfig();
  });
  [cfg.input, cfg.stride, cfg.padding].forEach((s) => s.addEventListener('change', applyConfig));

  // "Animate this in section 3" buttons
  document.querySelectorAll('[data-load]').forEach((btn) => btn.addEventListener('click', () => {
    const o = JSON.parse(btn.getAttribute('data-load'));
    cfg.input.value = o.input; cfg.kernel.value = o.kernel;
    cfg.stride.value = String(o.stride); cfg.padding.value = String(o.padding);
    applyConfig();
    $('animate').scrollIntoView({ behavior: 'smooth' });
    $('stepper').focus({ preventScroll: true });
  }));

  /* ---------------------------------------------------------------- 4A. cross-correlation vs convolution */
  const flipped = M.flip(W.kernel);
  const trueConv = M.convolve(W.input, W.kernel);
  U.selfCheck('True convolution', trueConv.output, D.concepts.trueConvolutionExpected);
  $('concept-a-kernels').append(
    grid(W.kernel, { label: 'K (used as-is by CNNs)' }),
    U.op('↻'),
    h('div', { class: 'muted', style: 'max-width:12em;font-size:.9rem', text: 'rotate 180°: reverse the rows, then reverse each row' }),
    U.op('→'),
    grid(flipped, { label: 'flip(K) (used by math convolution)' })
  );
  $('concept-a-outputs').append(
    h('div', { class: 'panel' }, [h('h4', { text: 'Cross-correlation (CNN "convolution")' }), grid(main.output, { label: 'I ⋆ K', size: 'lg' })]),
    h('div', { class: 'panel' }, [h('h4', { text: 'True convolution (kernel flipped)' }), grid(trueConv.output, { label: 'I ∗ K', size: 'lg' }),
      h('p', { class: 'muted', text: 'Every value has the opposite sign. Flipping this kernel left-to-right swaps +1 and −1, and flipping top-to-bottom changes nothing because all its rows are equal. So flip(K) = −K here.' })])
  );
  $('concept-a-step').appendChild(U.stepCard(trueConv, 0, { size: 'sm' }));

  /* ---------------------------------------------------------------- 4B. element-wise vs matrix multiplication */
  const s0 = main.steps[0];
  $('concept-b-good').appendChild(U.productEquation(s0, W.kernel, 'md'));
  $('concept-b-bad').appendChild(h('div', { class: 'eq' }, [
    grid(s0.region, { label: 'Region', size: 'sm' }), U.op('×'),
    grid(W.kernel, { label: 'Kernel', size: 'sm' }), U.op('='),
    grid(M.matmul(s0.region, W.kernel), { label: '✗ Matrix product: NOT used', size: 'md' })
  ]));

  /* ---------------------------------------------------------------- 4C. summation */
  const rowSums = s0.products.map((r) => r.reduce((a, b) => a + b, 0));
  $('concept-c-sum').innerHTML =
    '<div class="formula">' +
    '<div>All at once:  ' + M.sumExpression(s0.products) + ' = <b>' + fmt(s0.sum) + '</b></div>' +
    '<div style="padding-left:0">Row by row:  ' + s0.products.map((r, i) =>
      '(' + r.map(M.fmtTerm).join(' + ') + ') = ' + fmt(rowSums[i])).join(';  ') + '</div>' +
    '<div style="padding-left:0">then  ' + rowSums.map(M.fmtTerm).join(' + ') + ' = <b>' + fmt(s0.sum) + '</b></div>' +
    '</div>';
  const vec = (arr, label) => h('div', {}, [
    h('div', { class: 'matrix__label', html: label }),
    h('div', { class: 'vector' }, arr.map((v) => h('span', { text: fmt(v) })))
  ]);
  const vr = M.flatten(s0.region), vk = M.flatten(W.kernel);
  $('concept-c-dot').append(
    vec(vr, 'vec(region)'),
    vec(vk, 'vec(K)'),
    h('div', { class: 'formula', html: 'vec(region) · vec(K) = ' + vr.map((v, i) => fmt(v) + '·' + M.fmtTerm(vk[i])).join(' + ') + ' = <b>' + fmt(s0.sum) + '</b>' })
  );

  /* ---------------------------------------------------------------- 4D. stride */
  const s2 = M.crossCorrelate(W.input, W.kernel, { stride: 2 });
  U.selfCheck('Stride 2', s2.output, D.concepts.stride2Expected);
  const v1 = U.visitGrid(5, 5, 3, 0, 1, 'S = 1: window corners (9 positions)');
  const v2 = U.visitGrid(5, 5, 3, 0, 2, 'S = 2: window corners (4 positions)');
  $('concept-d-maps').append(
    h('div', { class: 'panel' }, [v1.grid.root, h('div', { html: U.formulaHTML(5, 3, 0, 1) })]),
    h('div', { class: 'panel' }, [v2.grid.root, h('div', { html: U.formulaHTML(5, 3, 0, 2) })])
  );
  $('concept-d-outputs').append(
    grid(main.output, { label: 'Feature map, S = 1 (3×3)', size: 'lg' }),
    grid(s2.output, { label: 'Feature map, S = 2 (2×2)', size: 'lg' })
  );
  s2.steps.forEach((_, k) => $('concept-d-steps').appendChild(U.stepCard(s2, k, { mini: true, size: 'sm' })));
  const uneven = U.visitGrid(6, 6, 3, 0, 2, '6×6 input, F = 3, S = 2');
  $('concept-d-uneven').append(
    h('div', {}, [uneven.grid.root]),
    h('div', { html: U.formulaHTML(6, 3, 0, 2) })
  );

  /* ---------------------------------------------------------------- 4E. padding */
  const p1 = M.crossCorrelate(W.input, W.kernel, { padding: 1 });
  U.selfCheck('Padding 1', p1.output, D.concepts.padding1Expected);
  $('concept-e-grids').append(
    h('div', { class: 'panel' }, [
      h('h4', { text: 'P = 0 (valid)' }),
      grid(W.input, { label: 'Input 5×5', size: 'sm' }),
      h('div', { html: U.formulaHTML(5, 3, 0, 1) }),
      grid(main.output, { label: 'Output 3×3', size: 'sm' })
    ]),
    h('div', { class: 'panel' }, [
      h('h4', { text: 'P = 1 (same)' }),
      grid(p1.padded, { label: 'Padded input 7×7 (grey = zeros)', size: 'sm', padding: 1 }),
      h('div', { html: U.formulaHTML(5, 3, 1, 1) }),
      grid(p1.output, { label: 'Output 5×5', size: 'sm' })
    ])
  );
  $('concept-e-step').appendChild(U.stepCard(p1, 0, { mini: true, size: 'sm' }));
  const p1s2 = M.crossCorrelate(W.input, W.kernel, { padding: 1, stride: 2 });
  U.selfCheck('Padding 1 stride 2', p1s2.output, D.concepts.padding1Stride2Expected);
  const v3 = U.visitGrid(5, 5, 3, 1, 2, 'Padded 7×7, window corners for S = 2');
  $('concept-e-both').append(
    h('div', { class: 'panel' }, [v3.grid.root]),
    h('div', { class: 'panel' }, [h('div', { html: U.formulaHTML(5, 3, 1, 2) }), grid(p1s2.output, { label: 'Output 3×3', size: 'md' }),
      h('p', { class: 'muted', text: 'These are the values at rows and columns 0, 2, 4 of the P = 1 output above, since stride 2 keeps every second window.' })])
  );

  /* ---------------------------------------------------------------- 4F. output size */
  const examples = [[5, 3, 0, 1], [5, 3, 0, 2], [5, 3, 1, 1], [5, 3, 1, 2], [6, 3, 0, 2], [28, 5, 0, 1], [32, 3, 1, 1], [224, 7, 3, 2], [227, 11, 0, 4]];
  let frows = '';
  examples.forEach(([N, F, P, S]) => {
    const st = M.outputSizeSteps(N, F, P, S);
    frows += '<tr><td>' + N + '</td><td>' + F + '</td><td>' + P + '</td><td>' + S + '</td>' +
      '<td class="mono">⌊(' + N + ' − ' + F + ' + ' + st.twoP + ') / ' + S + '⌋ + 1 = ⌊' + (+st.quotient.toFixed(3)) + '⌋ + 1</td>' +
      '<td><b>' + st.result + '×' + st.result + '</b></td></tr>';
  });
  $('concept-f-table').innerHTML = '<table class="vars"><tr><th>N</th><th>F</th><th>P</th><th>S</th><th>Substitution</th><th>Output</th></tr>' + frows + '</table>' +
    '<p class="muted">The last two rows are the first layers of ResNet (224, 7×7, S = 2) and AlexNet (227, 11×11, S = 4).</p>';
  new U.FormulaCalculator($('concept-f-calc'), { N: 5, F: 3, P: 0, S: 1 });

  /* ---------------------------------------------------------------- 4G. multiple filters */
  const K2 = D.concepts.horizontalKernel;
  const map2 = M.crossCorrelate(W.input, K2);
  U.selfCheck('Horizontal filter', map2.output, D.concepts.horizontalExpected);
  $('concept-g-maps').append(
    h('div', { class: 'eq' }, [grid(W.input, { label: 'Input I', size: 'sm' }), U.op('⊛'), grid(W.kernel, { label: 'Filter 1: vertical edge', size: 'sm' }), U.op('→'), grid(main.output, { label: 'Feature map 1', size: 'md' })]),
    h('div', { class: 'eq' }, [grid(W.input, { label: 'Input I', size: 'sm' }), U.op('⊛'), grid(K2, { label: 'Filter 2: horizontal edge', size: 'sm' }), U.op('→'), grid(map2.output, { label: 'Feature map 2', size: 'md' })])
  );
  $('concept-g-volume').append(
    h('div', { class: 'stack' }, [grid(main.output, { size: 'sm' }), grid(map2.output, { size: 'sm' })]),
    h('p', { html: 'Stacked: the layer output is a <b>3 × 3 × 2</b> volume. Slice 0 = map 1, slice 1 = map 2. The next layer sees both maps as 2 input channels.' })
  );
  // details: all 9 calculations for filter 2
  const det = h('details', { class: 'qa' }, [h('summary', { text: 'Show all 9 calculations for filter 2' })]);
  const detSteps = h('div', { class: 'steps' });
  map2.steps.forEach((_, k) => detSteps.appendChild(U.stepCard(map2, k, { mini: true, size: 'sm' })));
  det.appendChild(detSteps);
  $('concept-g-maps').appendChild(det);

  // Multi-channel: per-channel cross-correlation, then add channels + bias
  const mc = D.concepts.multiChannel;
  const multi = M.crossCorrelateMulti(mc.channels, mc.kernels, { bias: mc.bias });
  U.selfCheck('Multi-channel', multi.output, mc.expected);
  multi.perChannel.forEach((r, c) => U.selfCheck('Channel ' + (c + 1), r.output, mc.perChannelExpected[c]));
  const mcMount = $('concept-g-multi');
  multi.perChannel.forEach((r, c) => {
    mcMount.appendChild(h('div', { class: 'eq' }, [
      grid(mc.channels[c], { label: 'Channel ' + (c + 1), size: 'sm' }), U.op('⊛'),
      grid(mc.kernels[c], { label: 'Kernel slice ' + (c + 1), size: 'sm' }), U.op('→'),
      grid(r.output, { label: 'Channel ' + (c + 1) + ' result', size: 'sm' })
    ]));
  });
  const first = multi.perChannel.map((r) => r.steps[0]);
  mcMount.appendChild(h('div', { class: 'formula', html:
    '<div style="padding-left:0">Channel 1, position (0,0): ' + M.sumExpression(first[0].products) + ' = ' + fmt(first[0].sum) + '</div>' +
    '<div style="padding-left:0">Channel 2, position (0,0): ' + M.sumExpression(first[1].products) + ' = ' + fmt(first[1].sum) + '</div>' +
    '<div style="padding-left:0">O[0][0] = ' + fmt(first[0].sum) + ' + ' + fmt(first[1].sum) + ' + bias ' + fmt(mc.bias) + ' = <b>' + fmt(multi.output[0][0]) + '</b></div>' }));
  mcMount.appendChild(h('div', { class: 'eq' }, [
    grid(multi.perChannel[0].output, { label: 'Channel 1', size: 'sm' }), U.op('+'),
    grid(multi.perChannel[1].output, { label: 'Channel 2', size: 'sm' }), U.op('+'),
    h('div', { class: 'op', style: 'font-size:1.1rem', text: 'bias ' + fmt(mc.bias) }), U.op('='),
    grid(multi.output, { label: 'One feature map', size: 'md', className: 'matrix--final' })
  ]));

  /* ---------------------------------------------------------------- 6. practice */
  D.exercises.forEach((ex, i) => new U.Exercise($('exercises'), ex, i + 1));
  new U.SizeQuiz($('size-quiz'), D.sizeQuiz);
  new U.SizeQuiz($('filter-quiz'), D.filterQuiz);
}());
