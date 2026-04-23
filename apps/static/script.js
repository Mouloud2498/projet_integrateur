// =========================================================
// BOOKSMATCH — Frontend Script
// Fixes : segment mapping, erreurs robustes, toast modernisé
// =========================================================

// ─────────────────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────────────────
let currentUser = null;
let currentBookId = null;
let currentBookTitle = "";
let allBooks = [];
let likedIds = [];
let bookModal = null;

// ─────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const modalEl = document.getElementById("bookModal");
  if (modalEl) bookModal = new bootstrap.Modal(modalEl);
  bindKeyboard();
  bootstrapApp();
});

async function bootstrapApp() {
  await Promise.allSettled([loadHealth(), loadPopular(), loadCatalogue(), loadOnboardingBooks()]);
}

// ─────────────────────────────────────────────────────────
// UTILITAIRES
// ─────────────────────────────────────────────────────────
const qs = id => document.getElementById(id);

function esc(v) {
  return String(v ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function safeStr(v, fallback = "—") {
  return (v === null || v === undefined || v === "") ? fallback : String(v);
}

function toFrNumber(v) {
  try { return Number(v).toLocaleString("fr"); }
  catch { return String(v ?? "—"); }
}

function spinner() {
  return `<div class="spinner-wrapper"><div class="spinner-border" role="status"></div></div>`;
}

function emptyState(msg = "Aucun résultat disponible.") {
  return `<div class="empty-state"><div><i class="bi bi-book"></i><p>${esc(msg)}</p></div></div>`;
}

function coverUrl(id) { return `/api/covers/${id}`; }

async function fetchJSON(url, opts = {}) {
  const res = await fetch(url, opts);
  let payload = null;
  try { payload = await res.json(); } catch { }
  if (!res.ok) throw new Error(payload?.message || `Erreur ${res.status}`);
  return payload;
}

function normalizeGenre(raw) {
  const g = (raw || "").split("|")[0].trim();
  return g || "Littérature";
}

// Mapping segment serveur → label affiché + classe CSS
function segmentInfo(seg) {
  const map = {
    cold:       { label: "Découvreur", cls: "seg-cold" },
    warm:       { label: "Lecteur",    cls: "seg-warm" },
    hot:        { label: "Passionné",  cls: "seg-actif" },
    actif:      { label: "Passionné",  cls: "seg-actif" },
    decouvreur: { label: "Découvreur", cls: "seg-cold" },
    lecteur:    { label: "Lecteur",    cls: "seg-warm" },
    passionne:  { label: "Passionné",  cls: "seg-actif" },
  };
  return map[seg] || { label: seg || "Lecteur", cls: "seg-cold" };
}

function strategyBadge(meta = {}) {
  const alpha = Number(meta.alpha ?? 0);
  const n = meta.n_interactions || 0;

  if (!isNaN(alpha) && alpha >= 0.8)
    return `<span class="strategy-badge badge-hybrid">
      <i class="bi bi-stars"></i> Recommandations adaptées à votre profil
    </span>`;

  if (!isNaN(alpha) && alpha > 0)
    return `<span class="strategy-badge badge-content">
      <i class="bi bi-heart"></i> Basé sur vos goûts de lecture
    </span>`;

  return `<span class="strategy-badge badge-popular">
    <i class="bi bi-fire"></i> Tendances du moment
  </span>`;
}

function updateNavActive(section) {
  document.querySelectorAll(".nav-modern-link").forEach(el =>
    el.classList.toggle("active", el.dataset.section === section));
}

function showSection(name) {
  ["accueil", "catalogue", "onboarding"].forEach(s => {
    const el = qs(`section-${s}`);
    if (el) el.style.display = s === name ? "block" : "none";
  });
  updateNavActive(name);
  if (name === "accueil" && currentUser) loadPersonal();
  window.scrollTo({ top: 0, behavior: "smooth" });
}
window.showSection = showSection;

// ─────────────────────────────────────────────────────────
// TOASTS
// ─────────────────────────────────────────────────────────
function showToast(message, type = "success") {
  const zone = qs("toast-zone");
  if (!zone) return;
  const t = document.createElement("div");
  t.className = `toast-modern ${type}`;
  t.innerHTML = `<div>${esc(message)}</div>`;
  zone.appendChild(t);
  setTimeout(() => { t.style.opacity = "0"; t.style.transform = "translateX(10px)"; }, 3200);
  setTimeout(() => t.remove(), 3700);
}

// ─────────────────────────────────────────────────────────
// HEALTH
// ─────────────────────────────────────────────────────────
async function loadHealth() {
  try {
    const p = await fetchJSON("/health");
    const d = p.data || {};
    if (qs("stat-books")) qs("stat-books").textContent = toFrNumber(d.books_count ?? 0);
    if (qs("stat-users")) qs("stat-users").textContent = toFrNumber(d.users_count ?? 0);
  } catch (e) { console.warn("Health check failed:", e); }
}

// ─────────────────────────────────────────────────────────
// RENDER
// ─────────────────────────────────────────────────────────
function renderScoreLine(book, showScore) {
  const sim = book.similarity;
  const score = sim !== undefined && sim !== null ? Number(sim) : Number(book.score_normalise || 0);
  const label = sim !== undefined ? Number(sim).toFixed(2) : Number(book.score_normalise || 0).toFixed(2);
  if ((sim !== undefined && sim !== null) || (showScore && score > 0)) {
    const pct = Math.max(0, Math.min(100, score * 100));
    return `<div class="score-bar-wrap">
      <div class="score-bar"><div class="score-fill" style="width:${pct}%"></div></div>
      <span class="score-val">${label}</span>
    </div>`;
  }
  return "";
}

function renderCard(book, showScore = true, selectable = false) {
  const id = Number(book.book_id_mapping);
  const title = safeStr(book.title, "Titre inconnu");
  const safeTitle = esc(title);
  const genre = esc(normalizeGenre(book.genre || book.genre_clean || ""));
  const rating = book.rating_mean != null && !isNaN(Number(book.rating_mean))
    ? `<div class="book-rating"><i class="bi bi-star-fill"></i> ${Number(book.rating_mean).toFixed(1)}</div>` : "";
  const reason = book.reason
    ? `<div class="book-reason">${esc(book.reason)}</div>` : "";
  const author = book.author && book.author !== "unknown"
    ? `<div class="book-author"><i class="bi bi-person"></i> ${esc(book.author)}</div>` : "";
  const scoreLine = renderScoreLine(book, showScore);
  const isSelected = likedIds.includes(id);
  const selClass = selectable && isSelected ? "selected-liked" : "";
  const clickFn = selectable
    ? `toggleLiked(this, ${id})`
    : `openBook(${id}, '${safeTitle.replaceAll("&#39;", "\\'")}')`;

  return `
    <article class="book-card ${selClass}" onclick="${clickFn}" title="${safeTitle}">
      <div class="book-cover-wrap">
        <img class="book-cover" src="${coverUrl(id)}" alt="${safeTitle}" loading="lazy" />
        ${selectable
      ? `<div class="liked-check" id="chk-${id}" style="${isSelected ? "display:flex;" : ""}"><i class="bi bi-check-lg"></i></div>`
      : ""}
      </div>
      <div class="book-info">
        <div class="book-genre">${genre}</div>
        <div class="book-title">${safeTitle}</div>
        ${author}
        ${rating}${scoreLine}${reason}
      </div>
    </article>`;
}

function renderGrid(containerId, books, showScore = true, selectable = false, emptyMsg = "Aucun résultat disponible.") {
  const el = qs(containerId);
  if (!el) return;
  el.innerHTML = (!books || books.length === 0)
    ? emptyState(emptyMsg)
    : books.map(b => renderCard(b, showScore, selectable)).join("");
}

// ─────────────────────────────────────────────────────────
// POPULAIRE
// ─────────────────────────────────────────────────────────
async function loadPopular(genre = "") {
  const grid = qs("popular-grid");
  if (grid) grid.innerHTML = spinner();
  try {
    const url = genre
      ? `/api/recommend/popular?top_n=12&genre=${encodeURIComponent(genre)}`
      : "/api/recommend/popular?top_n=12";
    const p = await fetchJSON(url);
    renderGrid("popular-grid", p.data || [], false, false, "Aucun livre populaire pour ce filtre.");
  } catch (e) {
    if (grid) grid.innerHTML = emptyState("Impossible de charger les livres populaires.");
    showToast(e.message || "Chargement impossible.", "danger");
  }
}

function loadPopularByGenre() {
  loadPopular(qs("genre-filter-select")?.value || "");
}
window.loadPopularByGenre = loadPopularByGenre;

// ─────────────────────────────────────────────────────────
// RECOMMANDATIONS PERSONNELLES
// ─────────────────────────────────────────────────────────
async function loadPersonal() {
  if (!currentUser) return;
  const section = qs("personal-section");
  const grid = qs("personal-grid");
  const badge = qs("personal-strategy-badge");
  if (section) section.style.display = "block";
  if (grid) grid.innerHTML = spinner();
  try {
    const p = await fetchJSON("/api/recommend/user", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: currentUser.user_id, top_n: 10 })
    });
    if (badge) badge.innerHTML = strategyBadge(p.meta || {});
    renderGrid("personal-grid", p.data || [], true, false, "Aucune recommandation disponible.");
  } catch (e) {
    if (grid) grid.innerHTML = emptyState("Impossible de charger vos recommandations.");
    showToast(e.message || "Chargement impossible.", "danger");
  }
}



// ─────────────────────────────────────────────────────────
// AIDE
// ─────────────────────────────────────────────────────────
let helpModal = null;

function openHelp() {
  if (!helpModal) {
    const el = document.getElementById("helpModal");
    if (el) helpModal = new bootstrap.Modal(el);
  }
  helpModal?.show();
}

function switchHelp(btn, panelId) {
  document.querySelectorAll(".help-tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".help-panel").forEach(p => p.classList.remove("active"));
  btn.classList.add("active");
  const panel = document.getElementById(panelId);
  if (panel) panel.classList.add("active");
}

function switchHelpById(panelId) {
  const tabs = document.querySelectorAll(".help-tab");
  const panels = document.querySelectorAll(".help-panel");
  panels.forEach((p, i) => {
    if (p.id === panelId) {
      p.classList.add("active");
      tabs[i]?.classList.add("active");
    } else {
      p.classList.remove("active");
      tabs[i]?.classList.remove("active");
    }
  });
}

window.openHelp = openHelp;
window.switchHelp = switchHelp;
window.switchHelpById = switchHelpById;





async function sendContactForm() {
  const name    = qs("contact-name")?.value.trim();
  const email   = qs("contact-email")?.value.trim();
  const subject = qs("contact-subject")?.value.trim();
  const message = qs("contact-message")?.value.trim();

  if (!name || !email || !message) {
    showToast("Veuillez remplir tous les champs obligatoires.", "warning"); return;
  }
  if (!email.includes("@")) {
    showToast("Adresse email invalide.", "warning"); return;
  }

  const btn = document.querySelector("#help-contact .btn-hero-primary");
  if (btn) { btn.disabled = true; btn.innerHTML = `<i class="bi bi-hourglass-split"></i> Envoi...`; }

  try {
    await fetchJSON("/api/contact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, subject, message })
    });
    qs("contact-form-body").style.display = "none";
    qs("contact-success").style.display   = "flex";
    showToast("Message envoyé avec succès !", "success");
  } catch (e) {
    showToast("Erreur lors de l'envoi. Réessayez.", "danger");
    if (btn) { btn.disabled = false; btn.innerHTML = `<i class="bi bi-send-fill"></i> Envoyer le message`; }
  }
}
window.sendContactForm = sendContactForm;






// ─────────────────────────────────────────────────────────
// LOGIN / LOGOUT
// ─────────────────────────────────────────────────────────
async function loginUser() {
  const input = qs("user-id-input");
  const uid = parseInt(input?.value, 10);
  if (!uid || isNaN(uid)) { showToast("Entrez un numéro étudiant valide.", "warning"); return; }

  try {
    const p = await fetchJSON(`/api/users/${uid}`);
    const data = p.data || {};
    currentUser = { user_id: uid, ...data };

    qs("user-login-bar").style.display = "none";
    qs("user-badge").style.display = "flex";
    qs("badge-uid").textContent = `Utilisateur ${uid}`;

    const segEl = qs("badge-seg");
    const segData = segmentInfo(data.segment);
    segEl.textContent = segData.label;
    segEl.className = `badge-seg ${segData.cls}`;
    // ── Bannière progression ──────────────────────────────
    const n_int = data.n_interactions || 0;
    const MAX_BOOKS = 50;
    const pct = Math.min(n_int / MAX_BOOKS * 100, 100);
    if (qs("upb-name")) qs("upb-name").textContent = `Utilisateur ${uid}`;
    if (qs("upb-label")) qs("upb-label").textContent = `${n_int} livre${n_int > 1 ? "s" : ""} évalué${n_int > 1 ? "s" : ""}`;
    if (qs("upb-progress")) qs("upb-progress").style.width = pct + "%";
    if (qs("upb-progress-text")) qs("upb-progress-text").textContent = `${Math.round(pct)}% vers le profil complet (objectif : ${MAX_BOOKS} livres)`;
    // ─────────────────────────────────────────────────────
    showToast(`Bienvenue ! ${data.n_interactions || 0} interactions chargées.`, "success");
    showSection("accueil");
    await loadPersonal();
    setTimeout(() => {
      const s = qs("personal-section");
      if (s) s.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 300);
  } catch (e) {
    currentUser = null;
    showToast(e.message || "Utilisateur non trouvé.", "warning");
  }
}

function logoutUser() {
  currentUser = null;
  qs("user-login-bar").style.display = "flex";
  qs("user-badge").style.display = "none";
  const ps = qs("personal-section");
  if (ps) ps.style.display = "none";
  const inp = qs("user-id-input");
  if (inp) inp.value = "";
  showToast("Déconnecté.", "secondary");
}
window.loginUser = loginUser;
window.logoutUser = logoutUser;

// ─────────────────────────────────────────────────────────
// MODAL LIVRE
// ─────────────────────────────────────────────────────────
async function openBook(id, title) {
  currentBookId = id;
  currentBookTitle = title || "";
  if (!bookModal) return;

  // Réinitialisation
  qs("modal-cover").src = coverUrl(id);
  qs("modal-title").textContent = title || "Chargement...";
  qs("modal-genre").textContent = "...";
  qs("modal-rating").textContent = "";
  qs("modal-count").textContent = "";
  qs("modal-pages").textContent = "";
  qs("modal-desc").textContent = "Chargement...";
  if (qs("modal-pages-wrap")) qs("modal-pages-wrap").style.display = "";
  bookModal.show();

  try {
    const p = await fetchJSON(`/api/books/${id}`);
    const b = p.data || {};
    qs("modal-title").textContent = safeStr(b.title, title || "Livre");
    qs("modal-genre").textContent = normalizeGenre(b.genre_clean || b.genre || "");
    const authorEl = qs("modal-author");
    if (authorEl) authorEl.textContent = (b.author && b.author !== "unknown") ? b.author : "";
    if (authorEl) authorEl.style.display = (b.author && b.author !== "unknown") ? "" : "none";
    qs("modal-rating").textContent = b.rating_mean != null
      ? `${Number(b.rating_mean).toFixed(1)} / 5` : "—";
    qs("modal-count").textContent = b.rating_count
      ? `(${toFrNumber(parseInt(b.rating_count, 10))} évaluations)` : "";

    if (b.num_pages && Number(b.num_pages) > 0) {
      qs("modal-pages").textContent = parseInt(b.num_pages, 10);
    } else {
      if (qs("modal-pages-wrap")) qs("modal-pages-wrap").style.display = "none";
    }

    const desc = b.description && String(b.description).trim().length > 10
      ? String(b.description).replace(/\[.*?\]/g, "").replace(/\(.*?\)/g, "").trim()
      : "Aucune description disponible pour ce titre.";
    qs("modal-desc").textContent = desc;
  } catch (e) {
    qs("modal-desc").textContent = "Informations indisponibles.";
    console.error("openBook error:", e);
  }
}
window.openBook = openBook;

// ─────────────────────────────────────────────────────────
// LIVRES SIMILAIRES
// ─────────────────────────────────────────────────────────
async function loadSimilar(id, title) {
  if (bookModal) bookModal.hide();
  showSection('accueil');
  const section = qs("similar-section");
  const label = qs("similar-source-label");
  const grid = qs("similar-grid");
  if (section) section.style.display = "block";
  if (label) label.innerHTML = `<i class="bi bi-book-half me-2"></i>Livres similaires à <strong>${esc(title || "")}</strong>`;
  if (grid) grid.innerHTML = spinner();
  try {
    const p = await fetchJSON("/api/recommend/similar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ book_id_mapping: id, top_n: 8 })
    });
    renderGrid("similar-grid", p.data || [], true, false, "Aucun livre similaire trouvé.");
    section?.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    if (grid) grid.innerHTML = emptyState("Impossible de charger les livres similaires.");
    showToast(e.message || "Erreur.", "danger");
  }
}

function hideSimilar() {
  const s = qs("similar-section");
  if (s) s.style.display = "none";
}
window.loadSimilar = loadSimilar;
window.hideSimilar = hideSimilar;

// ─────────────────────────────────────────────────────────
// RECHERCHE
// ─────────────────────────────────────────────────────────
function handleSearch(e) { if (e.key === "Enter") doSearch(); }
function doSearch() {
  const q = qs("search-input")?.value?.trim() || "";
  if (!q) return;
  showSection("catalogue");
  if (qs("catalogue-search")) qs("catalogue-search").value = q;
  filterCatalogue();
}
window.handleSearch = handleSearch;
window.doSearch = doSearch;

// ─────────────────────────────────────────────────────────
// CATALOGUE
// ─────────────────────────────────────────────────────────
async function loadCatalogue() {
  const grid = qs("catalogue-grid");
  if (grid) grid.innerHTML = spinner();
  try {
    const p = await fetchJSON("/api/books?top_n=898");
    allBooks = p.data || [];
    buildGenreTags();
    buildSlideshow();
    filterCatalogue();
  } catch (e) {
    allBooks = [];
    if (grid) grid.innerHTML = emptyState("Impossible de charger le catalogue.");
  }
}

function filterCatalogue() {
  const q = (qs("catalogue-search")?.value || "").toLowerCase().trim();
  const genre = (qs("catalogue-genre")?.value || "").toLowerCase().trim();
  const sort = qs("catalogue-sort")?.value || "popularity";

  let filtered = [...allBooks].filter(b => {
    const tOk = !q ||
      (b.title || "").toLowerCase().includes(q) ||
      (b.author || "").toLowerCase().includes(q);
    const gOk = !genre || (b.genre_clean || "")
      .toLowerCase()
      .split("|")[0]
      .trim() === genre;
    return tOk && gOk;
  });

  if (sort === "rating")
    filtered.sort((a, b) => (Number(b.rating_mean) || 0) - (Number(a.rating_mean) || 0));
  else
    filtered.sort((a, b) => (Number(b.popularity_score) || 0) - (Number(a.popularity_score) || 0));

  if (qs("catalogue-count"))
    qs("catalogue-count").textContent = `${filtered.length} titre(s)`;

  renderGrid("catalogue-grid", filtered, false, false, "Aucun titre ne correspond.");
}
window.filterCatalogue = filterCatalogue;

// ─────────────────────────────────────────────────────────
// ONBOARDING
// ─────────────────────────────────────────────────────────
async function loadOnboardingBooks(genre = "") {
  const grid = qs("onboarding-grid");
  if (grid) grid.innerHTML = spinner();
  try {
    const url = genre ? `/api/books?top_n=50&genre=${encodeURIComponent(genre)}` : "/api/books?top_n=50";
    const p = await fetchJSON(url);
    renderGrid("onboarding-grid", p.data || [], false, true, "Aucun livre disponible.");
    restoreLikedVisuals();
  } catch (e) {
    if (grid) grid.innerHTML = emptyState("Impossible de charger les livres.");
  }
}

function restoreLikedVisuals() {
  likedIds.forEach(id => {
    const chk = qs(`chk-${id}`);
    document.querySelectorAll(".book-card").forEach(card => {
      if (card.getAttribute("onclick")?.includes(`toggleLiked(this, ${id})`))
        card.classList.add("selected-liked");
    });
    if (chk) chk.style.display = "flex";
  });
  updateLikedCount();
}




function closePersonal() {
  const s = qs("personal-section");
  if (s) s.style.display = "none";
}

function closeNewUserResults() {
  const s = qs("new-user-results");
  if (s) s.style.display = "none";
  // Reset sélection
  likedIds = [];
  updateLikedCount();
  document.querySelectorAll(".book-card.selected-liked").forEach(c => {
    c.classList.remove("selected-liked");
  });
  document.querySelectorAll(".liked-check").forEach(c => {
    c.style.display = "none";
  });
}

window.closePersonal = closePersonal;
window.closeNewUserResults = closeNewUserResults;



function toggleGenreChip(el) {
  document.querySelectorAll(".genre-chip").forEach(c => c.classList.remove("selected"));
  el.classList.add("selected");
  loadOnboardingBooks(el.dataset.genre || "");
}

function toggleLiked(card, id) {
  const idx = likedIds.indexOf(id);
  const chk = qs(`chk-${id}`);
  if (idx === -1) {
    likedIds.push(id);
    card.classList.add("selected-liked");
    if (chk) chk.style.display = "flex";
  } else {
    likedIds.splice(idx, 1);
    card.classList.remove("selected-liked");
    if (chk) chk.style.display = "none";
  }
  updateLikedCount();
}



// ─────────────────────────────────────────────────────────
// HERO SLIDESHOW
// ─────────────────────────────────────────────────────────
let slideIndex = 0;
let slideTimer = null;
let slideshowData = [];

function buildSlideshow() {
  const track = qs("slideshow-track");
  const dots = qs("slideshow-dots");
  if (!track || !allBooks.length) return;

  // Top livre par genre primaire
  const genreMap = {};
  allBooks.forEach(b => {
    const g = (b.genre_clean || "").split("|")[0].trim().toLowerCase();
    if (!g || g === "unknown") return;
    if (!genreMap[g] || (b.popularity_score || 0) > (genreMap[g].popularity_score || 0)) {
      genreMap[g] = b;
    }
  });

  const labelMap = {
    "fantasy": "Fantasy", "young-adult": "Jeunesse", "romance": "Romance",
    "fiction": "Fiction", "mystery": "Policier", "non-fiction": "Non-fiction",
    "comics": "Comics", "children": "Enfants", "poetry": "Poésie",
    "history": "Histoire", "historical fiction": "Fiction historique"
  };

  slideshowData = Object.entries(genreMap).slice(0, 10);

  track.innerHTML = slideshowData.map(([genre, book]) => {
    const label = labelMap[genre] || genre.charAt(0).toUpperCase() + genre.slice(1);
    const title = esc(book.title || "Titre inconnu");
    const author = book.author && book.author !== "unknown" ? esc(book.author) : "";
    const rating = book.rating_mean ? Number(book.rating_mean).toFixed(1) : "—";
    const count = book.rating_count ? `${toFrNumber(book.rating_count)} éval.` : "";
    const desc = book.description && book.description !== "unknown" && book.description.length > 10
      ? esc(book.description.substring(0, 160)) + "..."
      : "Découvrez ce livre populaire dans notre bibliothèque.";
    const bid = book.book_id_mapping;

    return `
      <div class="slide-item">
        <img class="slide-bg" src="/api/covers/${bid}" alt="${title}" />
        <div class="slide-gradient"></div>
        <img class="slide-cover" src="/api/covers/${bid}" alt="${title}" />
        <div class="slide-content">
          <div class="slide-genre-badge"><i class="bi bi-bookmark-fill"></i> ${label}</div>
          <div class="slide-title">${title}</div>
          ${author ? `<div class="slide-author"><i class="bi bi-person"></i> ${author}</div>` : ""}
          <div class="slide-meta">
            <div class="slide-meta-item"><i class="bi bi-star-fill"></i> ${rating}</div>
            ${count ? `<div class="slide-meta-item"><i class="bi bi-people-fill"></i> ${count}</div>` : ""}
            <div class="slide-meta-item"><i class="bi bi-tag-fill"></i> ${label}</div>
          </div>
          <div class="slide-desc">${desc}</div>
          <div class="slide-actions">
            <button class="slide-btn-primary" onclick="openBook(${bid}, '${title.replaceAll("'", "\\'")}')">
              <i class="bi bi-book-half"></i> Voir ce livre
            </button>
            <button class="slide-btn-secondary" onclick="filterByGenre('${genre}')">
              <i class="bi bi-grid"></i> Plus de ${label}
            </button>
          </div>
        </div>
      </div>`;
  }).join("");

  // Dots
  dots.innerHTML = slideshowData.map((_, i) =>
    `<button class="slide-dot ${i === 0 ? 'active' : ''}" onclick="goToSlide(${i})"></button>`
  ).join("");

  startSlideTimer();
}

function goToSlide(i) {
  slideIndex = i;
  const track = qs("slideshow-track");
  if (track) track.style.transform = `translateX(-${slideIndex * 100}%)`;
  document.querySelectorAll(".slide-dot").forEach((d, idx) =>
    d.classList.toggle("active", idx === slideIndex));
}

function slideNext() {
  goToSlide((slideIndex + 1) % slideshowData.length);
  resetSlideTimer();
}

function slidePrev() {
  goToSlide((slideIndex - 1 + slideshowData.length) % slideshowData.length);
  resetSlideTimer();
}

function startSlideTimer() {
  clearInterval(slideTimer);
  slideTimer = setInterval(slideNext, 5000);
}

function resetSlideTimer() {
  clearInterval(slideTimer);
  startSlideTimer();
}

window.slideNext = slideNext;
window.slidePrev = slidePrev;
window.goToSlide = goToSlide;










function buildGenreTags() {
  const wrap = document.querySelector(".genre-tags-wrap");
  const select = document.getElementById("catalogue-genre");
  if (!allBooks.length) return;

  const genres = {};
  allBooks.forEach(b => {
    const primary = (b.genre_clean || "").split("|")[0].trim().toLowerCase();
    if (primary && primary !== "unknown") {
      genres[primary] = (genres[primary] || 0) + 1;
    }
  });

  const sorted = Object.entries(genres).sort((a, b) => b[1] - a[1]);

  const labels = {
    "fantasy": "Fantasy",
    "young-adult": "Jeunesse",
    "romance": "Romance",
    "fiction": "Fiction",
    "mystery": "Policier",
    "non-fiction": "Non-fiction",
    "comics": "Comics",
    "children": "Enfants",
    "poetry": "Poésie",
    "history": "Histoire",
    "historical fiction": "Fiction historique",
    "biography": "Biographie",
    "paranormal": "Paranormal",
    "thriller": "Thriller",
    "crime": "Crime"
  };

  // ── Tags hero sidecard ───────────────────────────────────────
  if (wrap) {
    wrap.innerHTML = sorted.map(([genre, count]) => {
      const label = labels[genre] || genre.charAt(0).toUpperCase() + genre.slice(1);
      return `<span class="genre-tag" onclick="filterByGenre('${genre}')" title="${count} livre(s)">${label}</span>`;
    }).join("");
  }

  // ── Select catalogue ─────────────────────────────────────────
  if (select) {
    const current = select.value;
    select.innerHTML = `<option value="">Tous les genres</option>` +
      sorted.map(([genre]) => {
        const label = labels[genre] || genre.charAt(0).toUpperCase() + genre.slice(1);
        return `<option value="${genre}">${label}</option>`;
      }).join("");
    select.value = current;
  }
}

function filterByGenre(genre) {
  showSection('catalogue');
  setTimeout(() => {
    const genreInput = qs("catalogue-genre");
    if (genreInput) {
      genreInput.value = genre;
      filterCatalogue();
    }
  }, 80);
}

function updateLikedCount() {
  const n = likedIds.length;
  const lbl = qs("liked-count-label");
  const btn = qs("btn-get-recos");
  if (lbl) lbl.textContent = `${n} livre${n > 1 ? "s" : ""} sélectionné${n > 1 ? "s" : ""}`;
  if (btn) btn.disabled = false;
}

async function getNewUserRecos() {
  if (likedIds.length < 2) {
    showToast("Sélectionnez au moins 2 livres pour personnaliser vos recommandations.", "warning");
    const btn = qs("btn-get-recos");
    if (btn) {
      btn.classList.add("btn-shake");
      setTimeout(() => btn.classList.remove("btn-shake"), 600);
    }
    return;
  }
  const section = qs("new-user-results");
  const grid = qs("new-user-grid");
  if (section) section.style.display = "block";
  if (grid) grid.innerHTML = spinner();
  try {
    const p = await fetchJSON("/api/recommend/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ liked_ids: likedIds, top_n: 10 })
    });
    renderGrid("new-user-grid", p.data || [], true, false, "Aucune recommandation générée.");
    section?.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    if (grid) grid.innerHTML = emptyState("Impossible de générer les recommandations.");
    showToast(e.message || "Erreur.", "danger");
  }
}

window.loadOnboardingBooks = loadOnboardingBooks;
window.toggleGenreChip = toggleGenreChip;
window.toggleLiked = toggleLiked;
window.getNewUserRecos = getNewUserRecos;


// ─────────────────────────────────────────────────────────
// MODAL LIVRES LUS
// ─────────────────────────────────────────────────────────
let likedBooksModal = null;

async function openLikedBooks() {
  if (!currentUser) return;
  if (!likedBooksModal) {
    const el = document.getElementById("likedBooksModal");
    if (el) likedBooksModal = new bootstrap.Modal(el);
  }
  const container = qs("liked-books-list");
  if (container) container.innerHTML = spinner();
  likedBooksModal?.show();
  try {
    const p = await fetchJSON(`/api/users/${currentUser.user_id}`);
    const hist = (p.data?.history || []).sort((a, b) => b.rating - a.rating);
    if (!hist.length) {
      container.innerHTML = emptyState("Aucun livre dans l'historique.");
      return;
    }
    container.innerHTML = hist.map(b => {
      const stars = Math.round(b.rating || 0);
      const starStr = "★".repeat(stars) + "☆".repeat(5 - stars);
      const genre = (b.genre_clean || "").split("|")[0].trim() || "Littérature";
      const title = esc(b.title || "Titre inconnu");
      return `
        <div class="liked-book-row">
          <img class="liked-book-cover" src="/api/covers/${b.book_id_mapping}" alt="${title}" loading="lazy"/>
          <div class="liked-book-info">
            <div class="liked-book-title">${title}</div>
            <div class="liked-book-genre">${esc(genre)}</div>
          </div>
          <div class="liked-book-stars" title="${stars} / 5">${starStr}</div>
        </div>`;
    }).join("");
  } catch (e) {
    if (container) container.innerHTML = emptyState("Impossible de charger l'historique.");
  }
}
window.openLikedBooks = openLikedBooks;


// ─────────────────────────────────────────────────────────
// CLAVIER
// ─────────────────────────────────────────────────────────
function bindKeyboard() {
  const inp = qs("user-id-input");
  if (inp) inp.addEventListener("keydown", e => { if (e.key === "Enter") loginUser(); });
}

