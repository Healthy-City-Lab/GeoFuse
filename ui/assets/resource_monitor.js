// Resource monitor click handler.
//
// Loaded by ui/resource_monitor.py via ui/chrome.py:load_asset(). Lives in
// a zero-height iframe and reaches up via window.parent.document to flip
// body.gf-rm-open. Idempotent bindings (data-gfb attribute) so
// MutationObserver re-renders don't double-bind.
//
// Cross-browser notes:
//   - sessionStorage, classList.toggle, MutationObserver: baseline in every
//     browser we target. ES5-only syntax for safety on older Safari.

(function () {
	var D = window.parent.document;
	var KEY = "gf-rm-expanded";

	function apply() {
		var open = sessionStorage.getItem(KEY) === "1";
		D.body.classList.toggle("gf-rm-open", open);
	}

	function toggle(e) {
		if (e) e.stopPropagation();
		var open = !D.body.classList.contains("gf-rm-open");
		D.body.classList.toggle("gf-rm-open", open);
		sessionStorage.setItem(KEY, open ? "1" : "0");
	}

	function bind() {
		D.querySelectorAll(".gf-rm-mini:not([data-gfb])").forEach(function (el) {
			el.setAttribute("data-gfb", "1");
			el.addEventListener("click", toggle);
		});
	}

	if (!window.__gfRmObs) {
		window.__gfRmObs = new MutationObserver(function () {
			bind();
		});
		window.__gfRmObs.observe(D.body, { childList: true, subtree: true });
	}

	bind();
	apply();
})();
