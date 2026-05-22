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

	function moveTabsToToolbar() {
		var toolbar = d.querySelector("[data-testid='stToolbar']");
		var tabList = d.querySelector("[data-baseweb='tab-list']");
		if (!toolbar || !tabList) return;
		if (toolbar.contains(tabList)) return;
		// Streamlit's toolbar layout is:
		//   <stToolbar><div wrapper><div left-slot/><div right-slot/></div></stToolbar>
		// The left slot is intentionally empty -- drop the tab list into it
		// so Deploy / overflow stay on the right.
		var inner = toolbar.firstElementChild;
		if (!inner) return;
		var leftSlot = inner.firstElementChild;
		if (leftSlot) {
			leftSlot.appendChild(tabList);
		} else {
			inner.insertBefore(tabList, inner.firstChild);
		}
	}

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
		// Slot the toggle as the first child of the toolbar's left slot so
		// it sits to the left of the GeoFuse brand label and tabs.
		var toolbar = d.querySelector("[data-testid='stToolbar']");
		var inner = toolbar && toolbar.firstElementChild;
		var leftSlot = inner && inner.firstElementChild;
		var floater = d.querySelector(".gf-sidebar-floater");
		if (!leftSlot || !floater) return;
		if (floater.parentElement === leftSlot && floater === leftSlot.firstElementChild) return;
		leftSlot.insertBefore(floater, leftSlot.firstElementChild);
	}

	function bindSidebarToggle() {
		d.querySelectorAll(".gf-sidebar-floater:not([data-gfsb])").forEach(function (el) {
			el.setAttribute("data-gfsb", "1");
			el.addEventListener("click", function (e) {
				e.preventDefault();
				e.stopPropagation();
				toggleSidebar();
			});
			el.addEventListener("keydown", function (e) {
				if (e.key === "Enter" || e.key === " ") {
					e.preventDefault();
					toggleSidebar();
				}
			});
		});
	}

	function injectBrand() {
		var tabList = d.querySelector("[data-baseweb='tab-list']");
		if (!tabList) {
			setTimeout(injectBrand, 200);
			return;
		}
		if (tabList.querySelector(".gf-brand")) return;
		var span = d.createElement("span");
		span.className = "gf-brand";
		span.textContent = "GeoFuse";
		tabList.insertBefore(span, tabList.firstChild);
	}

	function markMutedTab() {
		var tabList = d.querySelector("[data-baseweb='tab-list']");
		if (!tabList) return;
		var tabs = tabList.querySelectorAll("button[role='tab']");
		for (var i = 0; i < tabs.length; i++) {
			tabs[i].classList.remove("gf-tab-muted");
		}
		if (tabs.length > 0) {
			tabs[tabs.length - 1].classList.add("gf-tab-muted");
		}
	}

	function apply() {
		moveTabsToToolbar();
		if (!d.querySelector("[data-baseweb='tab-list'] .gf-brand")) injectBrand();
		markMutedTab();
		ensureFloaterButton();
		placeFloater();
		bindSidebarToggle();
	}

	new MutationObserver(apply).observe(d.body || d.documentElement, { childList: true, subtree: true });

	apply();
})();
