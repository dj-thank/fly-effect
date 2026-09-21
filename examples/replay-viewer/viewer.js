(function () {
    "use strict";
    var data = window.REPLAY_DATA;
    if (!data || !data.meta) throw new Error("REPLAY_DATA is required");

    var n = data.meta.frames;
    var required = ["time_s", "contacts", "muscle_force", "cum_spikes", "cum_sensory", "root_xyz"];
    required.forEach(function (key) {
        if (!Array.isArray(data[key]) || data[key].length !== n) {
            throw new Error(key + " must contain exactly meta.frames entries");
        }
    });
    if (data.meta.pose_interpolation !== false) {
        throw new Error("Viewer accepts recorded, non-interpolated frames only");
    }

    var video = document.getElementById("video");
    var slider = document.getElementById("timeline");
    var play = document.getElementById("play");
    var current = -1;
    slider.max = String(Math.max(0, n - 1));

    function update(index) {
        index = Math.max(0, Math.min(n - 1, index));
        if (index === current) return;
        current = index;
        document.getElementById("time").textContent = data.time_s[index].toFixed(3) + " s";
        document.getElementById("spikes").textContent = data.cum_spikes[index].toLocaleString();
        document.getElementById("sensory").textContent = data.cum_sensory[index].toLocaleString();
        document.getElementById("force").textContent = data.muscle_force[index].toExponential(3);
        document.getElementById("root").textContent = data.root_xyz[index].map(function (v) {
            return v.toFixed(3);
        }).join(", ");
        data.meta.feet.forEach(function (foot, i) {
            document.getElementById("foot-" + foot).classList.toggle("on", data.contacts[index][i] === 1);
        });
        if (document.activeElement !== slider) slider.value = String(index);
    }

    function frameIndex() {
        return Math.floor(video.currentTime * data.meta.fps);
    }

    function tick() {
        update(frameIndex());
        window.requestAnimationFrame(tick);
    }

    play.addEventListener("click", function () {
        if (video.paused) video.play(); else video.pause();
    });
    video.addEventListener("play", function () { play.textContent = "Pause"; });
    video.addEventListener("pause", function () { play.textContent = "Play"; });
    slider.addEventListener("input", function () {
        var index = Number(slider.value);
        video.currentTime = (index + 0.01) / data.meta.fps;
        update(index);
    });
    document.getElementById("run").textContent = data.meta.run;
    document.getElementById("acceptance").textContent = data.meta.acceptance_passed ? "accepted" : "not accepted";
    update(0);
    tick();
})();
