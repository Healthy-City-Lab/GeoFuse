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
		if (!toolbar) return;

		var allTabLists = d.querySelectorAll("[data-baseweb='tab-list']");
		if (allTabLists.length === 0) return;

		// Prefer a tab-list that lives outside the toolbar — it is the one
		// React just rendered. The stale copy we moved on a prior run is
		// already inside the toolbar but is now orphaned from React's vDOM,
		// so React created a new one in the main body instead of updating it.
		var fresh = null;
		for (var i = 0; i < allTabLists.length; i++) {
			if (!toolbar.contains(allTabLists[i])) {
				fresh = allTabLists[i];
				break;
			}
		}
		if (!fresh) return; // all tab-lists already in toolbar — nothing to do

		// Remove stale copies left behind in the toolbar by a prior rerun.
		for (var j = 0; j < allTabLists.length; j++) {
			if (toolbar.contains(allTabLists[j])) {
				allTabLists[j].remove();
			}
		}

		var inner = toolbar.firstElementChild;
		if (!inner) return;
		var leftSlot = inner.firstElementChild;
		if (leftSlot) {
			leftSlot.appendChild(fresh);
		} else {
			inner.insertBefore(fresh, inner.firstChild);
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
		// Scope to the toolbar so we inject into the moved tab-list, not a
		// stale orphan or the pre-move copy in the main body.
		var toolbar = d.querySelector("[data-testid='stToolbar']");
		var tabList = toolbar && toolbar.querySelector("[data-baseweb='tab-list']");
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
		// Scope to the toolbar for the same reason as injectBrand.
		var toolbar = d.querySelector("[data-testid='stToolbar']");
		var tabList = toolbar && toolbar.querySelector("[data-baseweb='tab-list']");
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
		var toolbar = d.querySelector("[data-testid='stToolbar']");
		if (!toolbar || !toolbar.querySelector("[data-baseweb='tab-list'] .gf-brand")) injectBrand();
		markMutedTab();
		ensureFloaterButton();
		placeFloater();
		bindSidebarToggle();
	}

	new MutationObserver(apply).observe(d.body || d.documentElement, { childList: true, subtree: true });

	apply();
})();
