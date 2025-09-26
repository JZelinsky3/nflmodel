// tiny CSV parser (no deps). Assumes simple, unquoted CSV like your files.
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (!lines.length) return [];
  const headers = lines[0].split(",").map(h => h.trim());
  return lines.slice(1).map(line => {
    const cells = line.split(","); // simple split is fine for your CSVs
    const row = {};
    headers.forEach((h, i) => row[h] = (cells[i] ?? "").trim());
    return row;
  });
}

async function fetchCSV(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  const text = await res.text();
  return parseCSV(text);
}

function asPercent(x) {
  const v = parseFloat(x);
  if (!isFinite(v)) return NaN;
  return v > 1.000001 ? v : v * 100;
}

function pct(x) {
  if (x == null || x === "" || isNaN(+x)) return "";
  return `${(+x).toFixed(1)}%`;
}

function fmtOdds(american) {
  if (american == null || american === "" || isNaN(+american)) return "";
  const v = Math.round(+american);
  return v > 0 ? `+${v}` : `${v}`;
}

function pillHTML(valPct, bad=false) {
  const clamped = Math.max(0, Math.min(100, valPct));
  return `
    <div class="pill ${bad ? "bad": ""}" title="${clamped.toFixed(1)}%">
      <span style="width:${clamped}%;"></span>
    </div>`;
}

// ---------- Picks (pick_board_week_N.csv) ----------
let picksData = [];
let picksSortKey = "edge_num";
let picksSortDir = -1;

async function loadPicks(week) {
  try {
    const rows = await fetchCSV(`pick_board_week_${week}.csv`);
    picksData = rows.map(r => {
      const edgeNum   = asPercent(r["Edge %"] ?? r["edge %"] ?? r["edge"] ?? r["Edge"] ?? "");
      const modelPick = asPercent(r["Model% Pick"] ?? r["p_model_pick"] ?? r["model_pick_pct"] ?? "");
      const mktPick   = asPercent(r["Mkt% Pick"]   ?? r["p_market_pick"] ?? r["market_pick_pct"] ?? "");
      return {
        week: r["week"] || "",
        kickoff: r["kickoff"] || "",
        matchup: r["matchup"] || "",
        pick_side: r["pick_side"] || "",
        edge_num: edgeNum,
        model_pick_pct: modelPick,
        market_pick_pct: mktPick,
        fair_price_am: r["fair_price_am"] || "",
        spread_home: r["spread_home"] || r["Home Spread"] || "",
      };
      
    });
    document.getElementById("picks-note").classList.add("hidden");
    renderPicks();
  } catch (e) {
    picksData = [];
    document.getElementById("picks-table").querySelector("tbody").innerHTML = "";
    document.getElementById("picks-note").classList.remove("hidden");
  }
}

function renderPicks() {
  const q = document.getElementById("filter-picks").value.trim().toLowerCase();
  const rows = picksData
    .filter(r => {
      if (!q) return true;
      // support numeric filters like edge>3
      const m = q.match(/(edge|model|mkt)\s*([<>]=?)\s*([0-9.]+)/);
      if (m) {
        const [, key, op, valStr] = m;
        const val = parseFloat(valStr);
        const v = key === "edge" ? r.edge_num : key === "model" ? r.model_pick_pct : r.market_pick_pct;
        if (isNaN(v)) return false;
        if (op === ">") return v > val;
        if (op === "<") return v < val;
        if (op === ">=") return v >= val;
        if (op === "<=") return v <= val;
      }
      return (r.matchup.toLowerCase().includes(q) ||
              (r.pick_side||"").toLowerCase().includes(q));
    })
    .sort((a,b) => {
      const av = a[picksSortKey], bv = b[picksSortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return (av > bv ? 1 : av < bv ? -1 : 0) * picksSortDir;
    });

  const tbody = document.querySelector("#picks-table tbody");
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.week}</td>
      <td class="muted">${r.kickoff || ""}</td>
      <td>${r.matchup}</td>
      <td><span class="tag">${r.pick_side || ""}</span></td>
      <td style="min-width:120px;">
        ${pillHTML(Math.abs(r.edge_num), r.edge_num < 0)}
        <div class="muted">${r.edge_num ? r.edge_num.toFixed(1) + "%" : ""}</div>
      </td>
      <td style="min-width:120px;">
        ${pillHTML(r.model_pick_pct || 0)}
        <div class="muted">${isFinite(r.model_pick_pct) ? r.model_pick_pct.toFixed(1) + "%" : ""}</div>
      </td>
      <td style="min-width:120px;">
        ${pillHTML(r.market_pick_pct || 0, true)}
        <div class="muted">${isFinite(r.market_pick_pct) ? r.market_pick_pct.toFixed(1) + "%" : ""}</div>
      </td>
      <td>${fmtOdds(r.fair_price_am)}</td>
      <td>${r.spread_home ?? ""}</td>
    </tr>
  `).join("");
}

document.getElementById("load-week").addEventListener("click", () => {
  const w = +document.getElementById("week-input").value || 1;
  loadPicks(w);
});
document.getElementById("filter-picks").addEventListener("input", renderPicks);

document.querySelectorAll("#picks-table thead th").forEach(th => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    if (!key) return;
    if (picksSortKey === key) picksSortDir *= -1;
    else { picksSortKey = key; picksSortDir = -1; }
    renderPicks();
  });
});

// ---------- Upcoming (upcoming_with_features.csv) ----------
let slate = [];
let upSortKey = "week";
let upSortDir = 1;

async function loadUpcoming() {
  try {
    const rows = await fetchCSV("upcoming_with_features.csv");
    slate = rows.map(r => {
      // choose calibrated if present; else model; else market
      const pm = parseFloat(r["home_win_prob_market"] || r["p_market"] || "0") * 100;
      const pmod = r["home_win_prob_model_cal"] ? parseFloat(r["home_win_prob_model_cal"])*100 :
                    r["home_win_prob_model"] ? parseFloat(r["home_win_prob_model"])*100 : NaN;
      const edgeH = isFinite(pmod) ? (pmod - pm) : NaN;
      return {
        week: r["week"] || "",
        gameday: r["gameday"] || "",
        matchup: `${r["away_team"] || ""} @ ${r["home_team"] || ""}`,
        p_market_home: pm,
        p_model_home: pmod,
        edge_home: edgeH,
        spread_home: r["spread_home"] || "",
        total_line_close: r["total_line_close"] || ""
      };
    });
    renderUpcoming();
  } catch (e) {
    console.error("Failed to load upcoming_with_features.csv", e);
  }
}

function renderUpcoming() {
  const q = document.getElementById("filter-upcoming").value.trim().toLowerCase();
  const rows = slate
    .filter(r => {
      if (!q) return true;
      const m = q.match(/(edge|model)\s*([<>]=?)\s*([0-9.]+)/);
      if (m) {
        const [, key, op, valStr] = m;
        const v = key === "edge" ? r.edge_home : r.p_model_home;
        const val = parseFloat(valStr);
        if (!isFinite(v)) return false;
        if (op === ">") return v > val;
        if (op === "<") return v < val;
        if (op === ">=") return v >= val;
        if (op === "<=") return v <= val;
      }
      return (r.matchup.toLowerCase().includes(q));
    })
    .sort((a,b) => {
      const av = a[upSortKey], bv = b[upSortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return (av > bv ? 1 : av < bv ? -1 : 0) * upSortDir;
    });

  const tbody = document.querySelector("#upcoming-table tbody");
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.week}</td>
      <td class="muted">${r.gameday}</td>
      <td>${r.matchup}</td>
      <td style="min-width:110px;">
        ${pillHTML(r.p_market_home || 0, true)}
        <div class="muted">${isFinite(r.p_market_home) ? r.p_market_home.toFixed(1) + "%" : ""}</div>
      </td>
      <td style="min-width:110px;">
        ${pillHTML(isFinite(r.p_model_home) ? r.p_model_home : 0)}
        <div class="muted">${isFinite(r.p_model_home) ? r.p_model_home.toFixed(1) + "%" : ""}</div>
      </td>
      <td style="min-width:110px;">
        ${isFinite(r.edge_home) ? pillHTML(Math.abs(r.edge_home), r.edge_home<0) : ""}
        <div class="muted">${isFinite(r.edge_home) ? r.edge_home.toFixed(1) + "%" : ""}</div>
      </td>
      <td>${r.spread_home ?? ""}</td>
      <td>${r.total_line_close ?? ""}</td>
    </tr>
  `).join("");
}

document.querySelectorAll("#upcoming-table thead th").forEach(th => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    if (!key) return;
    if (upSortKey === key) upSortDir *= -1;
    else { upSortKey = key; upSortDir = -1; }
    renderUpcoming();
  });
});
document.getElementById("filter-upcoming").addEventListener("input", renderUpcoming);

// boot
loadUpcoming();
loadPicks(document.getElementById("week-input").value || 4);
