/*
 * components.js — reusable building blocks for the convolution lab.
 *
 *   MatrixGrid          draw any matrix; highlight a sliding window; update cells
 *   stepCard()          one filter position: region ⊙ kernel = products, Σ, store
 *   ConvolutionStepper  the animated filter (Next / Previous / Play / Pause / Reset)
 *   Exercise            a practice problem with hidden, step-by-step solution
 *   SizeQuiz            output-size questions
 *   FormulaCalculator   live Output Size = ⌊(N − F + 2P)/S⌋ + 1
 *   visitGrid()         where each window starts for a given stride/padding
 *   oneValueDiagram()   SVG: region × kernel → one output cell
 *
 * All arithmetic comes from ConvMath (conv-math.js); nothing here computes
 * convolution results on its own.
 */
(function (global) {
  'use strict';

  const M = global.ConvMath;
  const fmt = M.fmt;

  /* ---------- tiny DOM helper ---------- */
  function h(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null || v === false) continue;
        if (k === 'class') node.className = v;
        else if (k === 'html') node.innerHTML = v;
        else if (k === 'text') node.textContent = v;
        else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
        else node.setAttribute(k, v === true ? '' : v);
      }
    }
    for (const c of [].concat(children || [])) {
      if (c == null || c === false) continue;
      node.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    }
    return node;
  }

  function maxAbs(m) {
    let best = 0;
    for (const row of m) for (const v of row) if (typeof v === 'number') best = Math.max(best, Math.abs(v));
    return best || 1;
  }

  const plural = (n, word) => n + ' ' + word + (n === 1 ? '' : 's');

  /* Colour a cell by sign and size: blue = positive, red = negative. */
  function tintCell(cell, v, scale) {
    cell.classList.remove('pos', 'neg', 'zero');
    if (typeof v !== 'number') return;
    if (v === 0) { cell.classList.add('zero'); return; }
    cell.classList.add(v > 0 ? 'pos' : 'neg');
    cell.style.setProperty('--a', (0.10 + 0.45 * Math.min(1, Math.abs(v) / scale)).toFixed(3));
  }

  /* ======================================================================
   * MatrixGrid
   * options:
   *   label        caption (HTML allowed)
   *   size         'xs' | 'sm' | 'md' (default) | 'lg'
   *   tint         colour cells by value (default true)
   *   padding      P → the outer P rings are drawn as padding cells
   *   indices      show row/column numbers
   *   cellContent  (r, c, v) => string | Node   (custom cell body)
   *   blank        (r, c) => true to show an empty "?" cell
   *   window       {row, col, h, w} highlighted sliding window
   * ==================================================================== */
  class MatrixGrid {
    constructor(matrix, opts = {}) {
      this.matrix = matrix;
      this.opts = opts;
      this.rows = matrix.length;
      this.cols = matrix[0].length;
      this.off = opts.indices ? 1 : 0;                 // index labels take one row/col
      this.scale = opts.scale || maxAbs(matrix);

      this.root = h('figure', { class: 'matrix matrix--' + (opts.size || 'md') + (opts.className ? ' ' + opts.className : '') });
      if (opts.label) this.root.appendChild(h('figcaption', { class: 'matrix__label', html: opts.label }));

      this.grid = h('div', {
        class: 'matrix__grid',
        style: '--rows:' + (this.rows + this.off) + ';--cols:' + (this.cols + this.off),
        'aria-label': opts.ariaLabel || (this.rows + ' by ' + this.cols + ' matrix')
      });

      if (this.off) {
        this.grid.appendChild(h('div', { class: 'cell cell--index' }));
        for (let c = 0; c < this.cols; c++) this.grid.appendChild(h('div', { class: 'cell cell--index', text: c }));
      }

      this.cells = [];
      for (let r = 0; r < this.rows; r++) {
        if (this.off) this.grid.appendChild(h('div', { class: 'cell cell--index', text: r }));
        const rowCells = [];
        for (let c = 0; c < this.cols; c++) {
          const cell = h('div', { class: 'cell', 'data-r': r, 'data-c': c });
          const P = opts.padding || 0;
          if (P && (r < P || c < P || r >= this.rows - P || c >= this.cols - P)) cell.classList.add('cell--pad');
          rowCells.push(cell);
          this.grid.appendChild(cell);
          this.paint(cell, r, c, matrix[r][c]);
        }
        this.cells.push(rowCells);
      }

      this.windowEl = h('div', { class: 'matrix__window', hidden: true });
      this.grid.appendChild(this.windowEl);
      this.root.appendChild(h('div', { class: 'matrix__wrap' }, this.grid));
      if (opts.window) this.setWindow(opts.window);
    }

    setCell(r, c, v) {
      this.paint(this.cells[r][c], r, c, v);
    }

    paint(target, r, c, v) {
      const o = this.opts;
      if (o.blank && o.blank(r, c)) {
        target.textContent = '';
        target.classList.add('cell--blank');
        tintCell(target, null);
        return;
      }
      target.classList.remove('cell--blank');
      const content = o.cellContent ? o.cellContent(r, c, v) : fmt(v);
      target.textContent = '';
      if (typeof content === 'string') target.innerHTML = content;
      else target.appendChild(content);
      if (o.tint !== false) tintCell(target, v, this.scale);
    }

    /* Show a value now (used when the stepper fills the output map). */
    reveal(r, c, v) {
      const cell = this.cells[r][c];
      cell.classList.remove('cell--blank');
      cell.textContent = fmt(v);
      if (this.opts.tint !== false) tintCell(cell, v, this.scale);
    }

    hide(r, c) {
      const cell = this.cells[r][c];
      cell.textContent = '';
      cell.classList.add('cell--blank');
      tintCell(cell, null);
    }

    /* Move the highlighted window; CSS transitions make it slide. */
    setWindow(win) {
      if (!win) { this.windowEl.hidden = true; this.clearClass('in-window'); return; }
      this.windowEl.hidden = false;
      const s = this.windowEl.style;
      s.setProperty('--wr', win.row + this.off);
      s.setProperty('--wc', win.col + this.off);
      s.setProperty('--wh', win.h);
      s.setProperty('--ww', win.w);
      this.clearClass('in-window');
      for (let r = win.row; r < win.row + win.h; r++) {
        for (let c = win.col; c < win.col + win.w; c++) this.cells[r][c].classList.add('in-window');
      }
    }

    addClass(r, c, cls) { this.cells[r][c].classList.add(cls); }

    clearClass(cls) {
      for (const row of this.cells) for (const cell of row) cell.classList.remove(cls);
    }
  }

  /* An operator symbol placed between matrices: ⊙, =, →, +, … */
  function op(symbol, title) {
    return h('div', { class: 'op', title: title || null, 'aria-hidden': 'true', text: symbol });
  }

  /* Cell body for a products matrix: small "a×b" above the result. */
  function productCell(region, kernel) {
    return (r, c, v) => h('span', { class: 'prod' }, [
      h('span', { class: 'prod__calc', text: fmt(region[r][c]) + '×' + M.fmtTerm(kernel[r][c]) }),
      h('span', { class: 'prod__val', text: fmt(v) })
    ]);
  }

  /* Region ⊙ Kernel = Products, as three grids in a row. */
  function productEquation(step, kernel, size) {
    const sz = size || 'sm';
    return h('div', { class: 'eq' }, [
      new MatrixGrid(step.region, { label: 'Input region', size: sz }).root,
      op('⊙', 'element-wise multiplication'),
      new MatrixGrid(kernel, { label: 'Kernel', size: sz }).root,
      op('='),
      new MatrixGrid(step.products, {
        label: 'Products',
        size: sz === 'sm' ? 'md' : 'lg',
        className: 'matrix--products',
        cellContent: productCell(step.region, kernel)
      }).root
    ]);
  }

  function sumLine(step, bias) {
    let html = '<span class="sum__label">Sum</span> = ' + M.sumExpression(step.products) +
      ' = <b>' + fmt(step.sum) + '</b>';
    if (bias) html += ' &nbsp;then add bias ' + fmt(bias) + ' → <b>' + fmt(step.value) + '</b>';
    return h('p', { class: 'sumline', html: html });
  }

  /* Sentence that explains how the filter moves after this step. */
  function moveText(result, idx) {
    const steps = result.steps, S = result.stride;
    if (idx >= steps.length - 1) return 'That was the last valid position. The feature map is complete.';
    const cur = steps[idx], next = steps[idx + 1];
    if (next.outRow === cur.outRow) {
      return 'Next: slide right by S = ' + S + ' ' + (S === 1 ? 'column' : 'columns') +
        ' → the window starts at column ' + next.col + '.';
    }
    return 'End of this row (one more step right would push the window past the edge). Next: jump down by S = ' + S +
      ' ' + (S === 1 ? 'row' : 'rows') + ' to row ' + next.row + ' and return to column 0.';
  }

  function rangeText(step) {
    return 'rows ' + step.row + '–' + (step.row + step.h - 1) + ', columns ' + step.col + '–' + (step.col + step.w - 1);
  }

  /* ======================================================================
   * stepCard: the full calculation for one filter position.
   * opts.mini = true also draws the whole input with the window marked
   * and the output with the target cell marked.
   * ==================================================================== */
  function stepCard(result, idx, opts = {}) {
    const step = result.steps[idx];
    const n = result.steps.length;
    const padded = result.padding > 0;

    const card = h('article', { class: 'step' });
    card.appendChild(h('header', { class: 'step__head' }, [
      h('span', { class: 'step__num', text: 'Position ' + (idx + 1) + ' of ' + n }),
      h('span', { class: 'step__title', html: 'Output cell <b>O[' + step.outRow + '][' + step.outCol + ']</b>' }),
      h('span', { class: 'step__where', text: 'Window on ' + (padded ? 'padded ' : '') + 'input ' + rangeText(step) })
    ]));

    const body = h('div', { class: 'step__body' });
    if (opts.mini) {
      const inGrid = new MatrixGrid(result.padded, {
        label: padded ? 'Padded input' : 'Input', size: 'xs', padding: result.padding,
        window: { row: step.row, col: step.col, h: step.h, w: step.w }
      });
      const outGrid = new MatrixGrid(result.output, {
        label: 'Feature map', size: 'xs',
        blank: (r, c) => r * result.outW + c > idx
      });
      outGrid.addClass(step.outRow, step.outCol, 'is-current');
      body.appendChild(h('div', { class: 'step__mini' }, [inGrid.root, op('→'), outGrid.root]));
    }

    const calc = h('div', { class: 'step__calc' });
    calc.appendChild(h('p', { class: 'step__label', html: '<b>1–3.</b> Extract the region, multiply element by element:' }));
    calc.appendChild(productEquation(step, result.kernel, opts.size));
    calc.appendChild(h('p', { class: 'step__label', html: '<b>4.</b> Add all ' + (step.h * step.w) + ' products:' }));
    calc.appendChild(sumLine(step, result.bias));
    calc.appendChild(h('p', {
      class: 'step__store',
      html: '<b>5.</b> Store it: <span class="pill">O[' + step.outRow + '][' + step.outCol + '] = ' + fmt(step.value) + '</span>'
    }));
    calc.appendChild(h('p', { class: 'step__move', html: '<b>6.</b> ' + moveText(result, idx) }));
    body.appendChild(calc);
    card.appendChild(body);
    return card;
  }

  /* ======================================================================
   * Formula: Output Size = ⌊(N − F + 2P) / S⌋ + 1, with every step written out.
   * ==================================================================== */
  function formulaHTML(N, F, P, S, opts = {}) {
    const st = M.outputSizeSteps(N, F, P, S);
    const what = opts.what ? ' <span class="muted">(' + opts.what + ')</span>' : '';
    if (!st.valid) {
      return '<div class="formula"><div>⌊(' + N + ' − ' + F + ' + 2×' + P + ') / ' + S + '⌋ + 1</div>' +
        '<div class="warn">N − F + 2P = ' + st.numerator + ' is negative: the filter is bigger than the padded input, so there is no valid position.</div></div>';
    }
    const q = st.exact ? String(st.quotient) : String(+st.quotient.toFixed(3));
    let out = '<div class="formula">' +
      '<div>Output = ⌊(N − F + 2P) / S⌋ + 1' + what + '</div>' +
      '<div>= ⌊(' + N + ' − ' + F + ' + 2×' + P + ') / ' + S + '⌋ + 1</div>' +
      '<div>= ⌊(' + N + ' − ' + F + ' + ' + st.twoP + ') / ' + S + '⌋ + 1</div>' +
      '<div>= ⌊' + st.numerator + ' / ' + S + '⌋ + 1</div>' +
      '<div>= ⌊' + q + '⌋ + 1</div>' +
      '<div>= ' + st.floored + ' + 1 = <b>' + st.result + '</b></div>';
    if (!st.exact) {
      const Np = N + 2 * P;
      const unused = Np - ((st.result - 1) * S + F);
      out += '<div class="warn">' + st.numerator + ' is not divisible by ' + S + ', so the floor throws away the fraction. ' +
        'The last ' + plural(unused, 'row/column') + ' of the ' + (P ? 'padded ' : '') + 'input ' + (unused === 1 ? 'is' : 'are') +
        ' never covered by the filter.</div>';
    }
    return out + '</div>';
  }

  /* ======================================================================
   * visitGrid: a matrix the size of the (padded) input in which each
   * window's top-left corner is numbered in visiting order.
   * ==================================================================== */
  function visitGrid(rows, cols, F, P, S, label) {
    const v = M.visitMap(rows, cols, F, P, S);
    const g = new MatrixGrid(v.starts, {
      label: label, size: 'sm', tint: false, padding: P,
      cellContent: (r, c, n) => n ? '<b class="visit">' + n + '</b>' : ''
    });
    v.covered.forEach((row, r) => row.forEach((cov, c) => { if (!cov) g.addClass(r, c, 'cell--unused'); }));
    v.starts.forEach((row, r) => row.forEach((n, c) => { if (n) g.addClass(r, c, 'cell--start'); }));
    return { grid: g, info: v };
  }

  /* ======================================================================
   * ConvolutionStepper — the interactive sliding filter.
   * config: { input, kernel, stride, padding }
   * ==================================================================== */
  class ConvolutionStepper {
    constructor(mount, config) {
      this.mount = mount;
      this.delay = 1600;
      this.timer = null;
      mount.classList.add('stepper');
      mount.addEventListener('keydown', (e) => {
        if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
        if (e.key === 'ArrowRight') { this.next(); e.preventDefault(); }
        if (e.key === 'ArrowLeft') { this.prev(); e.preventDefault(); }
      });
      this.setConfig(config);
    }

    setConfig(cfg) {
      this.pause();
      this.cfg = cfg;
      this.mount.textContent = '';
      try {
        this.result = M.crossCorrelate(cfg.input, cfg.kernel, { stride: cfg.stride, padding: cfg.padding });
      } catch (err) {
        this.result = null;
        this.mount.appendChild(h('p', { class: 'warn', text: err.message }));
        return;
      }
      this.idx = 0;
      this.build();
      this.render();
    }

    build() {
      const R = this.result;
      const N = this.cfg.input.length, F = this.cfg.kernel.length;

      this.mount.appendChild(h('div', {
        class: 'stepper__formula',
        html: 'Input ' + N + '×' + this.cfg.input[0].length + ', kernel ' + F + '×' + this.cfg.kernel[0].length +
          ', S = ' + R.stride + ', P = ' + R.padding + ' → Output = ⌊(' + N + ' − ' + F + ' + 2×' + R.padding + ') / ' +
          R.stride + '⌋ + 1 = <b>' + R.outH + '</b>, so the feature map is <b>' + R.outH + '×' + R.outW + '</b> (' +
          R.steps.length + ' filter positions).'
      }));

      // Controls
      this.btn = {
        reset: h('button', { type: 'button', class: 'btn', onclick: () => this.reset(), text: '⟲ Reset' }),
        prev: h('button', { type: 'button', class: 'btn', onclick: () => this.prev(), text: '◀ Previous step' }),
        play: h('button', { type: 'button', class: 'btn btn--primary', onclick: () => this.play(), text: '▶ Play' }),
        pause: h('button', { type: 'button', class: 'btn', onclick: () => this.pause(), text: '❚❚ Pause' }),
        next: h('button', { type: 'button', class: 'btn btn--primary', onclick: () => this.next(), text: 'Next step ▶' })
      };
      const speed = h('select', { class: 'select', 'aria-label': 'Animation speed', onchange: (e) => {
        this.delay = +e.target.value;
        if (this.timer) { this.pause(); this.play(); }
      } }, [
        h('option', { value: 2600, text: 'Slow' }),
        h('option', { value: 1600, text: 'Normal', selected: true }),
        h('option', { value: 800, text: 'Fast' })
      ]);
      this.counter = h('span', { class: 'stepper__counter', 'aria-live': 'polite' });
      this.mount.appendChild(h('div', { class: 'controls' }, [
        this.btn.reset, this.btn.prev, this.btn.play, this.btn.pause, this.btn.next,
        h('label', { class: 'controls__speed' }, ['Speed ', speed]),
        this.counter
      ]));

      // Stage: input with sliding window, kernel, output being built
      this.inGrid = new MatrixGrid(R.padded, {
        label: R.padding ? 'Padded input (P = ' + R.padding + ')' : 'Input image I',
        indices: true, padding: R.padding, size: R.padded.length > 7 ? 'sm' : 'md'
      });
      this.kGrid = new MatrixGrid(this.cfg.kernel, { label: 'Kernel K', size: 'md' });
      this.outGrid = new MatrixGrid(R.output, {
        label: 'Feature map O', indices: true, size: 'md', blank: () => true
      });
      this.mount.appendChild(h('div', { class: 'stage' }, [
        this.inGrid.root, op('⊛', 'slide the kernel over the input'), this.kGrid.root, op('→'), this.outGrid.root
      ]));

      this.narration = h('p', { class: 'stepper__narration', 'aria-live': 'polite' });
      this.calc = h('div', { class: 'stepper__calc' });
      this.mount.appendChild(this.narration);
      this.mount.appendChild(this.calc);
    }

    render() {
      const R = this.result, step = R.steps[this.idx], n = R.steps.length;

      this.inGrid.setWindow({ row: step.row, col: step.col, h: step.h, w: step.w });

      // Output map: values up to the current step are known, the rest are still "?"
      this.outGrid.clearClass('is-current');
      R.steps.forEach((s, k) => {
        if (k <= this.idx) this.outGrid.reveal(s.outRow, s.outCol, s.value);
        else this.outGrid.hide(s.outRow, s.outCol);
      });
      this.outGrid.addClass(step.outRow, step.outCol, 'is-current');

      this.counter.textContent = 'Step ' + (this.idx + 1) + ' / ' + n;
      this.btn.prev.disabled = this.idx === 0;
      this.btn.next.disabled = this.idx === n - 1;
      this.btn.pause.disabled = !this.timer;
      this.btn.play.disabled = !!this.timer;

      const where = this.idx === 0 ? 'The filter starts in the top-left corner. ' : '';
      this.narration.innerHTML = where + 'Window covers ' + (R.padding ? 'padded ' : '') + 'input ' + rangeText(step) +
        ' → computes <b>O[' + step.outRow + '][' + step.outCol + ']</b>. ' + moveText(R, this.idx);

      this.calc.textContent = '';
      this.calc.appendChild(productEquation(step, R.kernel, 'sm'));
      this.calc.appendChild(sumLine(step, R.bias));
    }

    next() { if (this.idx < this.result.steps.length - 1) { this.idx++; this.render(); } else this.pause(); }
    prev() { if (this.idx > 0) { this.idx--; this.render(); } }
    reset() { this.pause(); this.idx = 0; this.render(); }

    play() {
      if (!this.result || this.timer) return;
      if (this.idx >= this.result.steps.length - 1) this.idx = 0;   // replay from the start
      this.timer = setInterval(() => {
        if (this.idx >= this.result.steps.length - 1) this.pause();
        else this.next();
      }, this.delay);
      this.render();
    }

    pause() {
      if (this.timer) clearInterval(this.timer);
      this.timer = null;
      if (this.result && this.btn) { this.btn.pause.disabled = true; this.btn.play.disabled = false; }
    }
  }

  /* ======================================================================
   * Exercise — try first, then check, then reveal the full working.
   * ==================================================================== */
  class Exercise {
    constructor(mount, ex, number) {
      this.ex = ex;
      this.result = M.crossCorrelate(ex.input, ex.kernel, { stride: ex.stride, padding: ex.padding });
      selfCheck('Exercise ' + ex.id, this.result.output, ex.expected);

      const card = h('article', { class: 'exercise', id: 'exercise-' + ex.id });
      card.appendChild(h('header', { class: 'exercise__head' }, [
        h('span', { class: 'badge badge--' + ex.level.toLowerCase().replace(/\W/g, ''), text: ex.level }),
        h('h3', { text: 'Exercise ' + number + ': ' + ex.title }),
        h('div', { class: 'tags' }, ex.tags.map((t) => h('span', { class: 'tag', text: t })))
      ]));
      card.appendChild(h('p', { text: ex.prompt }));
      card.appendChild(h('div', { class: 'eq' }, [
        new MatrixGrid(ex.input, { label: 'Input (' + ex.input.length + '×' + ex.input[0].length + ')', size: 'sm', indices: true }).root,
        op('⊛'),
        new MatrixGrid(ex.kernel, { label: 'Kernel (' + ex.kernel.length + '×' + ex.kernel[0].length + ')', size: 'sm' }).root,
        h('div', { class: 'params' }, [
          h('div', { html: 'Stride <b>S = ' + ex.stride + '</b>' }),
          h('div', { html: 'Padding <b>P = ' + ex.padding + '</b>' })
        ])
      ]));

      // Step A: output size (unlocks the answer grid)
      const rowsIn = h('input', { type: 'number', class: 'num', 'aria-label': 'Output rows', min: 1 });
      const colsIn = h('input', { type: 'number', class: 'num', 'aria-label': 'Output columns', min: 1 });
      const sizeMsg = h('span', { class: 'feedback', 'aria-live': 'polite' });
      const sizeBtn = h('button', { type: 'button', class: 'btn', text: 'Create answer grid', onclick: () => {
        if (+rowsIn.value === this.result.outH && +colsIn.value === this.result.outW) {
          sizeMsg.className = 'feedback ok';
          sizeMsg.textContent = 'Correct size. Now fill in every value.';
          this.buildAnswerGrid();
        } else {
          sizeMsg.className = 'feedback bad';
          sizeMsg.textContent = 'Not quite. Use ⌊(N − F + 2P)/S⌋ + 1 for rows and for columns.';
        }
      } });
      card.appendChild(h('div', { class: 'exercise__size' }, [
        h('span', { html: '<b>Step A.</b> Output size:' }), rowsIn, h('span', { text: '×' }), colsIn, sizeBtn, sizeMsg
      ]));

      // Step B: the answer grid
      this.answerMount = h('div', { class: 'exercise__answer' });
      card.appendChild(this.answerMount);

      this.feedback = h('p', { class: 'feedback', 'aria-live': 'polite' });
      this.hint = h('p', { class: 'hint', hidden: true, text: 'Hint: ' + ex.hint });
      this.solution = h('div', { class: 'solution', hidden: true });

      this.revealBtn = h('button', { type: 'button', class: 'btn', text: 'Reveal solution', onclick: () => this.toggleSolution() });
      card.appendChild(h('div', { class: 'controls' }, [
        h('button', { type: 'button', class: 'btn', text: 'Show hint', onclick: (e) => {
          this.hint.hidden = !this.hint.hidden;
          e.target.textContent = this.hint.hidden ? 'Show hint' : 'Hide hint';
        } }),
        this.revealBtn
      ]));
      card.appendChild(this.feedback);
      card.appendChild(this.hint);
      card.appendChild(this.solution);
      mount.appendChild(card);
    }

    buildAnswerGrid() {
      if (this.inputs) return;
      const R = this.result;
      this.inputs = [];
      const grid = h('div', { class: 'answer-grid', style: '--cols:' + R.outW });
      for (let r = 0; r < R.outH; r++) {
        const row = [];
        for (let c = 0; c < R.outW; c++) {
          const inp = h('input', { type: 'number', class: 'num num--cell', 'aria-label': 'O[' + r + '][' + c + ']', placeholder: 'O' + r + c });
          row.push(inp);
          grid.appendChild(inp);
        }
        this.inputs.push(row);
      }
      this.answerMount.appendChild(h('p', { html: '<b>Step B.</b> Your feature map (row by row):' }));
      this.answerMount.appendChild(grid);
      this.answerMount.appendChild(h('div', { class: 'controls' }, [
        h('button', { type: 'button', class: 'btn btn--primary', text: 'Check my answers', onclick: () => this.check() }),
        h('button', { type: 'button', class: 'btn', text: 'Clear', onclick: () => {
          this.inputs.flat().forEach((i) => { i.value = ''; i.classList.remove('ok', 'bad'); });
          this.feedback.textContent = '';
        } })
      ]));
    }

    check() {
      const R = this.result;
      let right = 0, filled = 0;
      this.inputs.forEach((row, r) => row.forEach((inp, c) => {
        inp.classList.remove('ok', 'bad');
        if (inp.value.trim() === '') return;
        filled++;
        const okay = Math.abs(parseFloat(inp.value) - R.output[r][c]) < 1e-9;
        inp.classList.add(okay ? 'ok' : 'bad');
        if (okay) right++;
      }));
      const total = R.outH * R.outW;
      this.feedback.className = 'feedback ' + (right === total ? 'ok' : 'bad');
      if (right === total) this.feedback.textContent = 'All ' + total + ' values correct. You computed the full feature map.';
      else if (filled < total) this.feedback.textContent = right + ' correct so far, ' + (total - filled) + ' still empty.';
      else this.feedback.textContent = right + ' of ' + total + ' correct. Red cells: recheck the products and the signs, then the sum.';
    }

    toggleSolution() {
      const show = this.solution.hidden;
      if (show && !this.solution.childElementCount) this.buildSolution();
      this.solution.hidden = !show;
      this.revealBtn.textContent = show ? 'Hide solution' : 'Reveal solution';
    }

    buildSolution() {
      const R = this.result, ex = this.ex;
      const s = this.solution;
      s.appendChild(h('h4', { text: 'Solution' }));
      s.appendChild(h('div', { html: formulaHTML(ex.input.length, ex.kernel.length, ex.padding, ex.stride, { what: 'rows' }) }));
      if (ex.input[0].length !== ex.input.length || ex.kernel[0].length !== ex.kernel.length) {
        s.appendChild(h('div', { html: formulaHTML(ex.input[0].length, ex.kernel[0].length, ex.padding, ex.stride, { what: 'columns' }) }));
      }
      if (ex.padding) {
        s.appendChild(h('p', { text: 'Pad with ' + ex.padding + ' ring of zeros first:' }));
        s.appendChild(new MatrixGrid(R.padded, { label: 'Padded input', size: 'sm', padding: ex.padding, indices: true }).root);
      }
      R.steps.forEach((_, k) => s.appendChild(stepCard(R, k, { mini: true, size: 'sm' })));
      s.appendChild(new MatrixGrid(R.output, { label: 'Final feature map', size: 'md', className: 'matrix--final' }).root);
    }
  }

  /* ======================================================================
   * SizeQuiz — numeric questions with a hidden worked answer.
   * item: { N, F, P, S, answer }  or  { question, answer, solution }
   * ==================================================================== */
  class SizeQuiz {
    constructor(mount, items) {
      const list = h('ol', { class: 'quiz' });
      items.forEach((it) => {
        const isFormula = it.N != null;
        if (isFormula) {
          const computed = M.outputSize(it.N, it.F, it.P, it.S);
          selfCheck('Size quiz N=' + it.N, [[computed]], [[it.answer]]);
        }
        const q = isFormula
          ? 'N = ' + it.N + ', F = ' + it.F + ', P = ' + it.P + ', S = ' + it.S + '. Output size?'
          : it.question;
        const inp = h('input', { type: 'number', class: 'num', 'aria-label': 'Answer' });
        const fb = h('span', { class: 'feedback', 'aria-live': 'polite' });
        const sol = h('div', { class: 'solution', hidden: true,
          html: isFormula ? formulaHTML(it.N, it.F, it.P, it.S) : '<p>' + it.solution + '</p>' });
        list.appendChild(h('li', {}, [
          h('p', { text: q }),
          h('div', { class: 'controls' }, [
            inp,
            h('button', { type: 'button', class: 'btn', text: 'Check', onclick: () => {
              const ok = +inp.value === it.answer && inp.value.trim() !== '';
              fb.className = 'feedback ' + (ok ? 'ok' : 'bad');
              fb.textContent = ok ? 'Correct.' : 'Not yet. Try again or reveal the working.';
            } }),
            h('button', { type: 'button', class: 'btn', text: 'Reveal working', onclick: (e) => {
              sol.hidden = !sol.hidden;
              e.target.textContent = sol.hidden ? 'Reveal working' : 'Hide working';
            } }),
            fb
          ]),
          sol
        ]));
      });
      mount.appendChild(list);
    }
  }

  /* ======================================================================
   * FormulaCalculator — type N, F, P, S and watch the substitution.
   * ==================================================================== */
  class FormulaCalculator {
    constructor(mount, init) {
      const vals = Object.assign({ N: 5, F: 3, P: 0, S: 1 }, init);
      const out = h('div', { class: 'calc__out', 'aria-live': 'polite' });
      const fields = {};
      const update = () => {
        const v = {};
        for (const k of Object.keys(fields)) v[k] = Math.floor(+fields[k].value);
        if (v.S < 1 || v.N < 1 || v.F < 1 || v.P < 0 || Object.values(v).some(Number.isNaN)) {
          out.innerHTML = '<p class="warn">Use N ≥ 1, F ≥ 1, P ≥ 0, S ≥ 1.</p>';
          return;
        }
        let html = formulaHTML(v.N, v.F, v.P, v.S);
        const r = M.outputSize(v.N, v.F, v.P, v.S);
        if (r >= 1) {
          html += '<p>An <b>' + v.N + '×' + v.N + '</b> input becomes a <b>' + r + '×' + r + '</b> feature map. ' +
            (r === v.N ? 'Same size as the input ("same" padding).' : r < v.N ? 'It shrank by ' + (v.N - r) + '.' : 'It grew by ' + (r - v.N) + '.') +
            ' The filter visits ' + r + ' × ' + r + ' = <b>' + (r * r) + '</b> positions.</p>';
        }
        out.innerHTML = html;
      };
      const labels = { N: 'N (input size)', F: 'F (filter size)', P: 'P (padding)', S: 'S (stride)' };
      const row = h('div', { class: 'calc__inputs' });
      for (const k of ['N', 'F', 'P', 'S']) {
        fields[k] = h('input', { type: 'number', class: 'num', value: vals[k], min: k === 'P' ? 0 : 1, oninput: update });
        row.appendChild(h('label', {}, [h('span', { text: labels[k] }), fields[k]]));
      }
      mount.appendChild(row);
      mount.appendChild(out);
      update();
    }
  }

  /* ======================================================================
   * oneValueDiagram — SVG version of "a window of the input times the
   * kernel produces one cell of the output" (recreates the idea of the
   * Stanford CS230 convolution-layer figure with real numbers).
   * ==================================================================== */
  function oneValueDiagram(result, idx) {
    const NS = 'http://www.w3.org/2000/svg';
    const s = (tag, attrs, text) => {
      const n = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
      if (text != null) n.textContent = text;
      return n;
    };
    const step = result.steps[idx];
    const C = 30;                                             // cell size in SVG units
    const inp = result.padded, ker = result.kernel, out = result.output;
    const inX = 10, inY = 120;
    const kX = 250, kY = 14;
    const oX = 470, oY = 120 + (inp.length - out.length) * C / 2;
    const width = oX + out[0].length * C + 12;
    const height = inY + inp.length * C + 30;
    const svg = s('svg', { viewBox: '0 0 ' + width + ' ' + height, class: 'diagram', role: 'img',
      'aria-label': 'The 3 by 3 window of the input is multiplied by the kernel and summed into output cell O[' + step.outRow + '][' + step.outCol + ']' });

    const drawGrid = (m, x0, y0, cls, mark) => {
      m.forEach((row, r) => row.forEach((v, c) => {
        const inMark = mark && r >= mark.row && r < mark.row + mark.h && c >= mark.col && c < mark.col + mark.w;
        svg.appendChild(s('rect', { x: x0 + c * C, y: y0 + r * C, width: C, height: C, class: 'd-cell ' + cls + (inMark ? ' d-hl' : '') }));
        const show = cls !== 'd-out' || r * out[0].length + c <= idx;
        if (show) svg.appendChild(s('text', { x: x0 + c * C + C / 2, y: y0 + r * C + C / 2 + 5, class: 'd-txt' }, fmt(v)));
      }));
    };

    // dashed projection lines from the window corners to the kernel corners
    const wx = inX + step.col * C, wy = inY + step.row * C, ww = step.w * C, wh = step.h * C;
    const kw = ker[0].length * C, kh = ker.length * C;
    [[wx, wy, kX, kY], [wx + ww, wy, kX + kw, kY], [wx, wy + wh, kX, kY + kh], [wx + ww, wy + wh, kX + kw, kY + kh]]
      .forEach(([x1, y1, x2, y2]) => svg.appendChild(s('line', { x1, y1, x2, y2, class: 'd-proj' })));

    drawGrid(inp, inX, inY, 'd-in', { row: step.row, col: step.col, h: step.h, w: step.w });
    svg.appendChild(s('rect', { x: wx, y: wy, width: ww, height: wh, class: 'd-window' }));
    drawGrid(ker, kX, kY, 'd-k');
    drawGrid(out, oX, oY, 'd-out', { row: step.outRow, col: step.outCol, h: 1, w: 1 });

    svg.appendChild(s('text', { x: inX, y: inY - 10, class: 'd-label' }, 'Input'));
    svg.appendChild(s('text', { x: kX + kw + 10, y: kY + kh / 2 + 5, class: 'd-label' }, 'Kernel'));
    svg.appendChild(s('text', { x: oX, y: oY - 10, class: 'd-label' }, 'Feature map'));

    // arrow from the window to the output cell, labelled with the result
    const ax1 = inX + inp[0].length * C + 14, ay = oY + step.outRow * C + C / 2, ax2 = oX - 8;
    svg.appendChild(s('line', { x1: ax1, y1: ay, x2: ax2, y2: ay, class: 'd-arrow' }));
    svg.appendChild(s('path', { d: 'M' + ax2 + ' ' + ay + ' l-9 -5 v10 z', class: 'd-arrowhead' }));
    svg.appendChild(s('text', { x: (ax1 + ax2) / 2, y: ay - 10, class: 'd-sum' }, 'Σ(region ⊙ K) = ' + fmt(step.value)));
    return svg;
  }

  /* Log an error if the engine ever disagrees with a stored answer. */
  function selfCheck(name, got, expected) {
    if (!expected) return true;
    const ok = JSON.stringify(got) === JSON.stringify(expected);
    if (!ok) console.error('[conv-lab] ' + name + ': computed', got, 'but expected', expected);
    return ok;
  }

  global.ConvUI = {
    h, op, MatrixGrid, stepCard, productEquation, sumLine, formulaHTML, visitGrid,
    ConvolutionStepper, Exercise, SizeQuiz, FormulaCalculator, oneValueDiagram, selfCheck
  };
}(window));
