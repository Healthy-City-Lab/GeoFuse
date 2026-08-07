// GeoFuse global page chrome — DOM observer.
//
// Loaded once per script run by ui/chrome.py:inject_chrome() inside a
// zero-height components.html iframe. From there it reaches up via
// window.parent.document to manipulate the actual Streamlit page DOM, and
// uses a MutationObserver to defend our edits against any React re-renders
// that would put the DOM back the way it was.
//
// New page-chrome JS belongs here, not in triple-quoted strings inside
// .py files. Keep operations idempotent so each apply() pass is safe.
//
// Cross-browser notes:
//   - Uses only ES5-safe syntax (var / function / no arrow funcs) so it
//     loads in older Safari/Firefox without a transpile step.
//   - MutationObserver, querySelector, classList, dispatchEvent: baseline
//     in every browser we target (Chromium, Firefox, Safari, Edge).

(function () {
	var d = window.parent.document;

	function toggleSidebar() {
		d.body.classList.toggle("gf-sb-hidden");
		// Streamlit recomputes some widget sizes on window resize. Triggering
		// one nudges st_folium, plots, etc. to fill the new width.
		window.parent.dispatchEvent(new Event("resize"));
	}

	function ensureFloaterButton() {
		// The primary render path is the <div role="button"> emitted via
		// st.markdown. This is a safety net in case the sanitiser strips it
		// -- only create one if nothing matches.
		if (d.querySelector(".gf-sidebar-floater")) return;
		if (!d.body) return;
		var el = d.createElement("div");
		el.className = "gf-sidebar-floater";
		el.setAttribute("role", "button");
		el.setAttribute("tabindex", "0");
		el.setAttribute("aria-label", "Toggle sidebar");
		el.title = "Toggle sidebar";
		d.body.appendChild(el);
	}

	function placeFloater() {
		// First child of the toolbar's left slot. Safe to move: this div is our
		// own markup and holds no React state.
		var toolbar = d.querySelector("[data-testid='stToolbar']");
		var inner = toolbar && toolbar.firstElementChild;
		var leftSlot = inner && inner.firstElementChild;
		if (!leftSlot) return;

		var allFloaters = d.querySelectorAll(".gf-sidebar-floater");
		if (allFloaters.length === 0) return;

		// Prefer a floater outside the left slot — it is the one Streamlit
		// just rendered. The stale copy (moved on a prior run) is in leftSlot
		// but orphaned from Streamlit's render tree.
		var fresh = null;
		for (var i = 0; i < allFloaters.length; i++) {
			if (allFloaters[i].parentElement !== leftSlot) {
				fresh = allFloaters[i];
				break;
			}
		}

		if (fresh) {
			// Evict stale copies from the left slot before inserting the fresh one.
			var stale = leftSlot.querySelectorAll(".gf-sidebar-floater");
			for (var j = 0; j < stale.length; j++) {
				stale[j].remove();
			}
			leftSlot.insertBefore(fresh, leftSlot.firstElementChild);
			return;
		}

		// All floaters are already in the left slot; make sure one is first.
		var existing = leftSlot.querySelector(".gf-sidebar-floater");
		if (existing && existing !== leftSlot.firstElementChild) {
			leftSlot.insertBefore(existing, leftSlot.firstElementChild);
		}
	}

	var _sidebarToggleBound = false;

	function bindSidebarToggle() {
		if (_sidebarToggleBound) return;
		_sidebarToggleBound = true;
		d.addEventListener("click", function (e) {
			var el = e.target;
			while (el && el !== d.body) {
				if (el.classList && el.classList.contains("gf-sidebar-floater")) {
					e.preventDefault();
					e.stopPropagation();
					toggleSidebar();
					return;
				}
				el = el.parentElement;
			}
		}, true);
		d.addEventListener("keydown", function (e) {
			if (e.key !== "Enter" && e.key !== " ") return;
			var el = e.target;
			while (el && el !== d.body) {
				if (el.classList && el.classList.contains("gf-sidebar-floater")) {
					e.preventDefault();
					toggleSidebar();
					return;
				}
				el = el.parentElement;
			}
		}, true);
	}

	// The tab list stays where React renders it. Moving it orphans the node
	// from React's vDOM, so the next rerender builds a fresh one -- resetting
	// the selection to the first tab. CSS places it instead; see chrome.css.
	function findTabList() {
		return d.querySelector("div[data-testid='stTabs'] [data-baseweb='tab-list']");
	}

	function injectBrand() {
		var list = findTabList();
		if (!list || list.querySelector(".gf-brand")) return;
		var span = d.createElement("span");
		span.className = "gf-brand";
		span.textContent = "GeoFuse";
		list.insertBefore(span, list.firstChild);
	}

	function markMutedTab() {
		var list = findTabList();
		if (!list) return;
		var tabs = list.querySelectorAll("button[role='tab']");
		for (var i = 0; i < tabs.length; i++) {
			tabs[i].classList.remove("gf-tab-muted");
		}
		if (tabs.length > 0) {
			tabs[tabs.length - 1].classList.add("gf-tab-muted");
		}
	}

	function apply() {
		injectBrand();
		markMutedTab();
		ensureFloaterButton();
		placeFloater();
		bindSidebarToggle();
	}

	new MutationObserver(apply).observe(d.body || d.documentElement, { childList: true, subtree: true });

	apply();
})();
