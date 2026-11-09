// Layout audit for the guided tour. Serve web/ and this file from the same origin, open any page there,
// then: await import("/layout-audit.js"); await audit(375). Reports overflow, clipped content and wrapping labels.
window.audit = async function audit(W, H = 820) {
  document.body.innerHTML = "";
  document.body.style.margin = "0";
  const f = document.createElement("iframe");
  f.style.cssText = `width:${W}px;height:${H}px;border:0`;
  f.src = "/?audit=" + Date.now();
  document.body.append(f);
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  await new Promise((r) => (f.onload = r));
  await wait(2500);
  const d = f.contentDocument;
  const issues = [];
  const name = (e) => `${e.tagName.toLowerCase()}${e.id ? "#" + e.id : ""}${e.classList.length ? "." + [...e.classList].join(".") : ""}`;
  const txt = (e) => (e.textContent || "").trim().slice(0, 40);
  const PILLS = ".truth-chip,.status,.verdict,.btn,.key,.eyebrow,.jbrand,.jlink,.go,.linkish,.meter-val,.switch .who,.compare div,.mini-bar span,.meter-name,.dots,.path h3";
  const CONTAINERS = ".visual,.path,.switch,.card,.mini,.big-digit,.keypad,.meter,.ring,.why,.answer > div";

  function check(screen) {
    const win = f.contentWindow;
    const vw = d.documentElement.clientWidth;
    if (d.documentElement.scrollWidth > vw + 1) issues.push([screen, "page scrolls sideways", `${d.documentElement.scrollWidth} > ${vw}`]);
    const all = d.querySelectorAll(".jbar *, .jstage *, .mini, .mini *");
    for (const e of all) {
      if (e.closest("svg") && e.tagName.toLowerCase() !== "svg") continue;
      const r = e.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      const cs = win.getComputedStyle(e);
      if (cs.position === "fixed") continue;
      if (r.right > vw + 0.5 || r.left < -0.5) issues.push([screen, "outside viewport", `${name(e)} "${txt(e)}" ${Math.round(r.left)}..${Math.round(r.right)}`]);
      // content clipped or spilling inside a box
      if (e.scrollWidth > e.clientWidth + 2 && e.clientWidth > 0 && !["svg", "canvas", "img"].includes(e.tagName.toLowerCase())) {
        if (cs.overflowX === "auto" || cs.overflowX === "scroll") continue;
        issues.push([screen, "content wider than box", `${name(e)} "${txt(e)}" ${e.scrollWidth} > ${e.clientWidth}`]);
      }
    }
    for (const e of d.querySelectorAll(PILLS)) {
      const r = e.getBoundingClientRect();
      if (!r.height) continue;
      const lh = parseFloat(win.getComputedStyle(e).lineHeight) || parseFloat(win.getComputedStyle(e).fontSize) * 1.3;
      const rects = [...e.childNodes].filter((n) => n.nodeType === 3 && n.textContent.trim()).flatMap((n) => {
        const range = d.createRange();
        range.selectNodeContents(n);
        return [...range.getClientRects()];
      });
      const lines = new Set(rects.map((q) => Math.round(q.top))).size;
      if (lines > 1) issues.push([screen, "label wraps", `${name(e)} "${txt(e)}" (${lines} lines)`]);
    }
    for (const c of d.querySelectorAll(CONTAINERS)) {
      const cr = c.getBoundingClientRect();
      for (const e of c.querySelectorAll("*")) {
        if (e.closest("svg") && e.tagName.toLowerCase() !== "svg") continue;
        const r = e.getBoundingClientRect();
        if (!r.width || !r.height) continue;
        if (r.right > cr.right + 1 || r.left < cr.left - 1 || r.bottom > cr.bottom + 1)
          issues.push([screen, "spills out of container", `${name(e)} "${txt(e)}" out of ${name(c)}`]);
      }
    }
  }

  const click = (sel) => d.querySelector(sel)?.click();
  check("draw empty");
  click("#example");
  await wait(700);
  check("draw filled");
  click("#done");
  await wait(2600);
  check("choose");
  const order = ["signal", "mind", "break"];
  for (let k = 0; k < 3; k++) {
    const j = order[k];
    if (k === 0) click(`[data-j=${j}]`);
    await wait(1300);
    for (let step = 1; step < 9 && !d.querySelector(".keypad"); step++) {
      check(`${j} step ${step}`);
      if (j === "break" && d.querySelector(".switch")) {
        for (const s of d.querySelectorAll(".switch")) s.click();
        await wait(500);
        check(`${j} step ${step} all off`);
      }
      click(".nav .btn");
      await wait(1300);
    }
    check(`${j} guess`);
    d.querySelectorAll(".key")[3]?.click();
    await wait(3600);
    check(`${j} reveal`);
    const next = order[k + 1];
    if (next) {
      [...d.querySelectorAll(".next-ways .btn")].find((b) => b.textContent.includes({ mind: "Read its mind", break: "Try to break it" }[next]))?.click();
    }
  }
  const uniq = [...new Map(issues.map((i) => [i.join("|"), i])).values()];
  window.__audit = window.__audit || {};
  window.__audit[W] = uniq;
  return { width: W, count: uniq.length, issues: uniq.slice(0, 40) };
};
