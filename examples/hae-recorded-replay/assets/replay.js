(function () {
  'use strict';
  var D = window.REPLAY_DATA;
  var P = window.REPLAY_PROVENANCE;
  if (!P || P.recording_kind !== 'simulated_recorded_run' ||
      P.calibration_status !== 'uncalibrated' || P.changes_made !== true ||
      P.source_recording_included !== false || P.raw_inputs_included !== false) {
    throw new Error('Replay provenance boundary is missing or invalid');
  }
  var N = D.meta.frames;
  var FPS = D.meta.fps;
  var video = document.getElementById('video');
  var slider = document.getElementById('timeline');
  var playBtn = document.getElementById('play');
  var timeEl = document.getElementById('time');
  var spikesEl = document.getElementById('spikes');
  var sensoryEl = document.getElementById('sensory');
  var forceEl = document.getElementById('force');
  var rootEl = document.getElementById('root');
  var spark = document.getElementById('spark');
  var sctx = spark.getContext('2d');

  var footEls = {};
  D.meta.feet.forEach(function (f) {
    footEls[f] = document.getElementById('foot-' + f);
  });

  var fmin = Math.min.apply(null, D.muscle_force);
  var fmax = Math.max.apply(null, D.muscle_force);
  var current = -1;

  function drawSpark(i) {
    var w = spark.width, h = spark.height, pad = 8;
    sctx.clearRect(0, 0, w, h);
    sctx.strokeStyle = '#2557a7';
    sctx.lineWidth = 1.2;
    sctx.beginPath();
    for (var k = 0; k < N; k++) {
      var x = pad + (w - 2 * pad) * k / (N - 1);
      var y = h - pad - (h - 2 * pad) * (D.muscle_force[k] - fmin) / (fmax - fmin || 1);
      if (k === 0) sctx.moveTo(x, y); else sctx.lineTo(x, y);
    }
    sctx.stroke();
    if (i >= 0) {
      var xi = pad + (w - 2 * pad) * i / (N - 1);
      sctx.strokeStyle = '#c0392b';
      sctx.lineWidth = 1;
      sctx.beginPath();
      sctx.moveTo(xi, pad);
      sctx.lineTo(xi, h - pad);
      sctx.stroke();
    }
  }

  function update(i) {
    if (i === current) return;
    current = i;
    timeEl.textContent = D.time_s[i].toFixed(2) + ' / ' + D.meta.duration_s.toFixed(2) + ' s';
    D.meta.feet.forEach(function (f, k) {
      footEls[f].classList.toggle('on', D.contacts[i][k] === 1);
    });
    spikesEl.textContent = D.cum_spikes[i].toLocaleString() + ' / ' + D.meta.total_spikes.toLocaleString();
    sensoryEl.textContent = D.cum_sensory[i].toLocaleString() + ' / ' + D.meta.total_sensory.toLocaleString();
    forceEl.textContent = D.muscle_force[i].toExponential(3);
    var p = D.root_xyz[i];
    rootEl.textContent = p[0].toFixed(3) + ', ' + p[1].toFixed(3) + ', ' + p[2].toFixed(3);
    if (document.activeElement !== slider) slider.value = i;
    drawSpark(i);
  }

  function frameIndex() {
    return Math.min(N - 1, Math.max(0, Math.floor(video.currentTime * FPS)));
  }

  function tick() {
    update(frameIndex());
    requestAnimationFrame(tick);
  }

  playBtn.addEventListener('click', function () {
    if (video.paused) { video.play(); } else { video.pause(); }
  });
  video.addEventListener('play', function () { playBtn.textContent = '一時停止'; });
  video.addEventListener('pause', function () { playBtn.textContent = '再生'; });
  video.addEventListener('ended', function () { playBtn.textContent = '再生'; });

  document.querySelectorAll('button.speed').forEach(function (b) {
    b.addEventListener('click', function () {
      video.playbackRate = parseFloat(b.dataset.rate);
      document.querySelectorAll('button.speed').forEach(function (x) { x.classList.remove('active'); });
      b.classList.add('active');
    });
  });

  slider.addEventListener('input', function () {
    video.currentTime = (parseInt(slider.value, 10) + 0.01) / FPS;
    update(parseInt(slider.value, 10));
  });

  video.addEventListener('loadedmetadata', function () { update(0); });
  drawSpark(-1);
  tick();
})();
