"use strict";

/* =========================================================
   後端位址（自動沿用目前頁面的 http/https，避免混合內容）
   ========================================================= */
const HOST  = location.hostname || "127.0.0.1";
const PROTO = (location.protocol === "https:") ? "https" : "http";

// 5000：偵測演算法主機（/result 仍走這台）
const API_BASE = (HOST === "localhost" || HOST === "127.0.0.1")
  ? `${PROTO}://127.0.0.1:5000`
  : `${PROTO}://${HOST}:5000`;

// 5001：浮水印主機
const WM_API_BASE = (HOST === "localhost" || HOST === "127.0.0.1")
  ? `${PROTO}://127.0.0.1:5001`
  : `${PROTO}://${HOST}:5001`;

// 27042 報告同源（app.py）；實際路徑未知時做 fallback
const ORIGIN_BASE = `${location.origin.replace(/\/$/, "")}`;

/* ===================== 小工具 ===================== */
function $(id){ return document.getElementById(id); }
function showSpinner(id, text){
  const el=$(id);
  if(!el) return;
  if(typeof text === "string") el.textContent = text;
  el.style.display="block";
}
function hideSpinner(id){ const el=$(id); if(el) el.style.display="none"; }
function displayResponse(id, msg, isError=false){
  const el=$(id); if(!el) return;
  el.textContent = msg || "";
  el.className = "response " + (isError ? "error" : (msg ? "success" : ""));
}
function previewImage(inputId, previewId){
  const fi=$(inputId), pv=$(previewId); if(!fi||!pv) return;
  fi.addEventListener("change",(e)=>{
    const f=e.target.files?.[0];
    if(f && f.type?.startsWith("image/")){
      const r=new FileReader();
      r.onload=(ev)=>{ pv.src=ev.target.result; pv.style.display="block"; pv.classList.add("zoomable"); };
      r.readAsDataURL(f);
    }else{
      pv.src=""; pv.style.display="none";
    }
  });
}
function setText(id, text){
  const el=$(id); if(!el) return false; el.textContent = text ?? ""; el.style.display = text ? "" : "none"; return true;
}
function setLink(id, href, label){
  const a=$(id); if(!a) return false;
  if(href && /^https?:\/\//i.test(href)){
    a.href = href; a.textContent = label || href; a.target = "_blank"; a.rel = "noopener"; a.style.display="";
  }else{
    a.removeAttribute("href"); a.textContent = ""; a.style.display="none";
  }
  return true;
}
async function copyToClipboard(text){
  try{ await navigator.clipboard.writeText(text); return true; }catch(_){ return false; }
}
const delay = (ms)=> new Promise(res=> setTimeout(res, ms));

/* 轉成完整 WM URL（支援 /files/...） */
function absoluteWMUrl(path){
  if(!path) return null;
  if (/^(https?:|data:)/i.test(path)) return path;
  if (path.startsWith("/")) return `${WM_API_BASE}${path}`;
  return pathToOpenImageURL(path);
}

/* 新增：把可能是相對路徑的 URL 補成同源完整 URL */
function normalizeMaybeRelativeURL(u){
  if(!u) return null;
  if(/^https?:\/\//i.test(u) || u.startsWith("data:")) return u;
  if(u.startsWith("/")) return `${ORIGIN_BASE}${u}`;
  return u;
}

/* ===================== 導覽下拉 ===================== */
function toggleMenu(){
  const menu=$("dropdownMenu"); if(!menu) return;
  menu.style.display = (menu.style.display==="block"?"none":"block");
}
document.addEventListener("click",(e)=>{
  const menu=$("dropdownMenu"); const btn=document.querySelector(".dropbtn");
  if(!menu||!btn) return;
  if(!menu.contains(e.target) && !btn.contains(e.target)) menu.style.display="none";
});

/* === 拿掉 nav 裡的 .reveal，避免被全域樣式設為透明 === */
function stripRevealInNav(){
  const nav = document.querySelector("nav");
  if(!nav) return;
  nav.querySelectorAll(".reveal").forEach(el=>{
    el.classList.remove("reveal");
    el.removeAttribute("data-anim");
    el.style.opacity = "";
    el.style.filter  = "";
    el.style.transform = "";
  });
  nav.querySelectorAll(".dropdown-content a").forEach(a=>{
    a.style.opacity   = "1";
    a.style.filter    = "none";
    a.style.transform = "none";
    a.style.display   = "block";
    a.style.color     = "#1f2d3d";
    a.style.pointerEvents = "auto";
  });
}

/* ===================== 左右拖曳對比（照出真身用） ===================== */
function ensureCompareMount(containerId, leftId, rightId){
  const box = $(containerId);
  const left = $(leftId);
  const right = $(rightId);
  if(!box || !left || !right) return;

  Object.assign(box.style, {
    position: "relative",
    overflow: "hidden",
    touchAction: "none",
    userSelect: "none",
  });

  [left, right].forEach(img=>{
    Object.assign(img.style, {
      display: "block",
      width: "100%",
      height: "auto",
      objectFit: "contain"
    });
  });

  right.style.position = "absolute";
  right.style.left = "0";
  right.style.top = "0";
  right.style.clipPath = "inset(0 0 0 50%)";

  let handle = box.querySelector(".cmp-handle");
  if(!handle){
    handle = document.createElement("div");
    handle.className = "cmp-handle";
    Object.assign(handle.style, {
      position: "absolute",
      top: "0",
      bottom: "0",
      width: "2px",
      background: "rgba(255,255,255,.9)",
      boxShadow: "0 0 0 1px rgba(0,0,0,.4)",
      cursor: "ew-resize",
      left: "50%"
    });
    const knob = document.createElement("div");
    Object.assign(knob.style, {
      position: "absolute",
      top: "50%", left: "-10px",
      width: "20px", height: "20px",
      marginTop: "-10px",
      borderRadius: "50%",
      background: "rgba(0,0,0,.65)"
    });
    handle.appendChild(knob);
    box.appendChild(handle);
  }

  const updateAt = (clientX)=>{
    const r = box.getBoundingClientRect();
    let x = Math.min(Math.max(clientX - r.left, 0), r.width);
    const pct = (x / r.width) * 100;
    right.style.clipPath = `inset(0 0 0 ${100 - pct}%)`;
    handle.style.left = `${pct}%`;
  };

  setTimeout(()=>{
    const r = box.getBoundingClientRect();
    updateAt(r.left + r.width/2);
  }, 0);

  let dragging = false;
  const start = (e)=>{ dragging = true; e.preventDefault(); };
  const move  = (e)=>{
    if(!dragging) return;
    if(e.touches && e.touches[0]) updateAt(e.touches[0].clientX);
    else updateAt(e.clientX);
  };
  const end   = ()=>{ dragging = false; };

  handle.addEventListener("mousedown", start);
  document.addEventListener("mousemove", move);
  document.addEventListener("mouseup", end);
  handle.addEventListener("touchstart", start, {passive:false});
  document.addEventListener("touchmove", move, {passive:false});
  document.addEventListener("touchend", end, {passive:true});

  handle.setAttribute("tabindex","0");
  handle.addEventListener("keydown", (e)=>{
    const step = (e.shiftKey ? 5 : 1);
    const r = box.getBoundingClientRect();
    const cur = parseFloat(handle.style.left || "50") || 50;
    if(e.key === "ArrowLeft"){ updateAt(r.left + (Math.max(0, cur - step) / 100) * r.width); }
    if(e.key === "ArrowRight"){ updateAt(r.left + (Math.min(100, cur + step) / 100) * r.width); }
  });

  window.addEventListener("resize", ()=>{
    const r = box.getBoundingClientRect();
    updateAt(r.left + (parseFloat(handle.style.left || "50")/100)*r.width);
  }, {passive:true});
}

/* ===================== 解碼用：URL 解析（5001） ===================== */
function pathToOpenImageURL(path){
  if(!path) return null;
  if(/^(https?:|data:)/i.test(path)) return path;

  // Flask /files/ 靜態路徑
  if(path.startsWith("/files/")){
    return `${WM_API_BASE}${path}`;
  }

  try{
    // Unix：.../forensics_runs/<case>/<name>.png
    const m1 = path.match(/\/forensics_runs\/([^/]+)\/([^/]+\.png)/i);
    if(m1) return `${WM_API_BASE}/open/image/${encodeURIComponent(m1[1])}/${encodeURIComponent(m1[2])}`;
    // Windows：...\forensics_runs\<case>\<name>.png
    const m2 = path.match(/\\forensics_runs\\([^\\]+)\\([^\\]+\.png)/i);
    if(m2) return `${WM_API_BASE}/open/image/${encodeURIComponent(m2[1])}/${encodeURIComponent(m2[2])}`;
  }catch(_){}

  return `${WM_API_BASE}/open/image?path=${encodeURIComponent(path)}`;
}

/* === 後端回傳的最佳 reveal URL 選擇 === */
function pickBestRevealURL(data){
  if (data?.images?.tile_best_256){
    const u = data.images.tile_best_256;
    if (typeof u === 'string'){
      return (/^(https?:|data:)/i.test(u)) ? u : pathToOpenImageURL(u);
    }
  }

  if(data?.best_reveal_dataurl){
    const x = data.best_reveal_dataurl;
    if (typeof x === 'string'){
      if (x.startsWith('data:') || /^https?:\/\//i.test(x)) return x;
      return pathToOpenImageURL(x);
    }
  }
  const toURL = (x) => {
    if(!x) return null;
    if(typeof x !== "string") return null;
    if (x.startsWith("data:") || /^https?:\/\//i.test(x)) return x;
    return pathToOpenImageURL(x);
  };

  if (data?.best_pick) {
    const pick = String(data.best_pick).toLowerCase();
    if (pick.includes("tile_best_x4")) {
      return toURL(data.images?.tile_best_x4) || toURL(data.tile_best_x4_dataurl);
    }
    if (pick.includes("tile_best")) {
      return toURL(data.images?.tile_best) || toURL(data.tile_best_dataurl);
    }
    if (pick.includes("reveal")) {
      return toURL(data.images?.revealed) || toURL(data.images?.reveal) ||
             toURL(data.reveal_dataurl) ||
             (data.s2_prime_b64 ? `data:image/png;base64,${data.s2_prime_b64}` : null);
    }
  }

  const candidates = [
    data.best_reveal_dataurl,
    data.tile_best_x4_dataurl,
    data.images?.tile_best_x4,
    data.tile_best_dataurl,
    data.images?.tile_best,
    data.reveal_dataurl || (data.s2_prime_b64 ? `data:image/png;base64,${data.s2_prime_b64}` : null),
    data.images?.revealed,
    data.images?.reveal,
    data.reveal
  ];

  for(const c of candidates){
    const u = toURL(c);
    if(u) return u;
  }
  return null;
}

/* ======== 下載工具 ======== */
async function forceDownload(url, filename){
  if (/^data:/i.test(url)) {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename || "download";
    document.body.appendChild(a);
    a.click();
    a.remove();
    return;
  }
  try{
    const resp = await fetch(url, { mode: "cors", credentials: "include" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const blob = await resp.blob();
    const obj = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = obj;
    a.download = filename || (url.split("/").pop() || "download");
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(obj);
  }catch(_){
    window.open(url, "_blank", "noopener");
  }
}
function bindDownloadLink(anchorId, href, filename){
  const el = $(anchorId);
  if (!el || !href) return;
  el.style.display = "inline-block";
  el.href = href;
  el.removeAttribute("download");
  el.removeAttribute("target");
  el.removeAttribute("rel");
  const clone = el.cloneNode(true);
  el.parentNode.replaceChild(clone, el);
  clone.addEventListener("click", async (e)=>{
    e.preventDefault();
    await forceDownload(toDownloadURL(href), filename);
  });
}
function bindOpenInNewTab(anchorId, href, label){
  const el = $(anchorId);
  if (!el || !href) return;
  el.style.display = "inline-block";
  el.href = href;
  el.textContent = label || el.textContent || "開啟";
  el.target = "_blank";
  el.rel = "noopener";
  const clone = el.cloneNode(true);
  el.parentNode.replaceChild(clone, el);
}

/* 下載時將 /files/ 轉成完整 URL */
function toDownloadURL(url){
  try{
    if(!url) return url;
    if(/^data:/i.test(url)) return url;
    if(url.startsWith("/files/")) return `${WM_API_BASE}${url}`;
    const m = url.match(/\/open\/image\/([^/]+)\/([^/]+\.[a-zA-Z0-9]+)/i);
    if(m) return `${WM_API_BASE}/dl/image/${encodeURIComponent(m[1])}/${encodeURIComponent(m[2])}`;
  }catch(_){ }
  return url;
}

/* ===================== 施法加印（5001） ===================== */
function _getPreferNorm(){
  const viaId = $("preferNorm")?.value?.trim().toUpperCase();
  const viaRadio = document.querySelector('input[name="preferNorm"]:checked')?.value?.trim().toUpperCase();
  const v = viaRadio || viaId;
  return (v==="BN"||v==="IN") ? v : null;
}
function _appendCollectionFields(fd){
  const m = {
    collection_interface: $("collectionInterface")?.value,
    collection_device: $("collectionDevice")?.value,
    write_blocker: $("writeBlocker")?.value,
    collection_steps: $("collectionSteps")?.value,
    collection_deviations: $("collectionDeviations")?.value,
  };
  Object.entries(m).forEach(([k,v])=>{
    if(typeof v === "string" && v.trim()!=="") fd.append(k, v.trim());
  });
}

async function runEmbedGeneric(coverInputId, secretInputId){
  const cover  = $(coverInputId)?.files?.[0];
  const secret = secretInputId ? ($(secretInputId)?.files?.[0] || null) : null;
  if(!cover){ displayResponse("wmEmbedResp","請先選擇一張圖片（Cover）。",true); return; }
  if(secretInputId && !secret){
    displayResponse("wmEmbedResp","此模式需要 Cover＋Secret 兩張圖。若只想單檔嵌入，請用單檔按鈕。",true);
    return;
  }

  const gridCtrl = $("secretGrid");
  const gridVal = gridCtrl?.value && /^\d+$/.test(gridCtrl.value) ? parseInt(gridCtrl.value,10) : null;
  const makeDelivery = !!$("makeDelivery")?.checked;
  const preferNorm = _getPreferNorm();

  showSpinner("wmEmbedSpinner","處理中…");
  displayResponse("wmEmbedResp","");

  try{
    const fd = new FormData();
    fd.append("cover", cover);
    if(secret) fd.append("secret", secret);
    if(gridVal && [2,3,4].includes(gridVal)) fd.append("secret_grid", String(gridVal));
    if(makeDelivery) fd.append("make_delivery", "true");
    if(preferNorm) fd.append("prefer_norm", preferNorm);
    _appendCollectionFields(fd);

    const resp = await fetch(`${WM_API_BASE}/infer27037`, { method:"POST", body:fd, mode:"cors" });
    if(!resp.ok){
      let t=""; try{ t=await resp.text(); }catch(_){}
      throw new Error(`HTTP ${resp.status}${t?`: ${t.substring(0,200)}`:""}`);
    }

    const ct = (resp.headers.get("content-type")||"").toLowerCase();

    if(ct.startsWith("image/")){
      const blob = await resp.blob();
      const url  = URL.createObjectURL(blob);
      updateWMUI({ container_dataurl:url }, cover.name);
      displayResponse("wmEmbedResp","完成 ✔️");
    }else{
      const data = await resp.json();

      if(data?.files){
        if(data.files.container) data.container_dataurl = absoluteWMUrl(data.files.container);
        if(data.files.secret)    data.secret_dataurl    = absoluteWMUrl(data.files.secret);
        if(data.files.report)    data.report            = absoluteWMUrl(data.files.report);
      }

      if(!data.container_dataurl && data.images?.container){
        data.container = pathToOpenImageURL(data.images.container);
      }
      if(!data.reveal_dataurl && data.images?.revealed){
        data.reveal_dataurl = pathToOpenImageURL(data.images.revealed);
      }
      if(!data.secret_dataurl && data.images?.secret_in){
        data.secret_dataurl = pathToOpenImageURL(data.images.secret_in);
      }
      if(data.images?.tile_best){
        data.images.tile_best = pathToOpenImageURL(data.images.tile_best);
      }
      if(data.images?.tile_best_x4){
        data.images.tile_best_x4 = pathToOpenImageURL(data.images.tile_best_x4);
      }

      updateWMUI(data, cover.name);
      updateWMReportUI(data);
      updateTileMetaUI(data);

      const bestUrl = pickBestRevealURL(data);
      if(bestUrl){
        const stem = (cover.name||"image").replace(/\.[^.]+$/,"");
        bindDownloadLink("downloadBestReveal", bestUrl, `${stem}_revealed.png`);
        bindDownloadLink("downloadRevealed",    bestUrl, `${stem}_revealed.png`);
      }

      displayResponse("wmEmbedResp","完成 ✔️");
    }
  }catch(err){
    let hint="無法連線 Watermark 後端（5001）。";
    if(location.protocol==="https:" && WM_API_BASE.startsWith("http://"))
      hint += "（HTTPS↔HTTP 混合內容被瀏覽器擋下）";
    displayResponse("wmEmbedResp",`${hint} 詳細：${err.message}`,true);
  }finally{
    hideSpinner("wmEmbedSpinner");
  }
}

function updateWMUI(data, originalName="image"){
  const containerUrl = data.container_dataurl
    || data.container
    || (data.files?.container ? absoluteWMUrl(data.files.container) : null)
    || (data.images?.container ? pathToOpenImageURL(data.images.container) : null);

  const revealedUrl = pickBestRevealURL(data);

  const tiledUrl     = data.secret_dataurl
    || (data.files?.secret ? absoluteWMUrl(data.files.secret) : null)
    || data.secret_tiled_dataurl
    || (data.images?.secret_in ? pathToOpenImageURL(data.images.secret_in) : null);

  const cImg = $("imgContainer");
  if(cImg && containerUrl){ cImg.src = containerUrl; cImg.classList.add("zoomable"); cImg.style.display="block"; }

  const rImg = $("imgRevealed");
  if(rImg && revealedUrl){ rImg.src = revealedUrl; rImg.classList.add("zoomable"); rImg.style.display="block"; }

  const sImg = $("imgSecretTiled") || $("imgSecret");
  if(sImg && tiledUrl){ sImg.src = tiledUrl; sImg.classList.add("zoomable"); sImg.style.display="block"; }

  const stem = (originalName||"image").replace(/\.[^.]+$/,"");
  if(containerUrl){
    bindDownloadLink("downloadLink1",     containerUrl, `${stem}_wm.png`);
    bindDownloadLink("downloadContainer", containerUrl, `${stem}_wm.png`);
  }
  if(tiledUrl){
    bindDownloadLink("downloadSecret",    tiledUrl,     `secret_${Date.now()}.png`);
  }
  if(revealedUrl){
    bindDownloadLink("downloadBestReveal", revealedUrl, `${stem}_revealed.png`);
    bindDownloadLink("downloadRevealed",   revealedUrl, `${stem}_revealed.png`);
  }

  const wrap = $("imageResults1");
  if(wrap) wrap.style.display="block";

  const mWrap = $("wmMetrics");
  if(mWrap){
    const a = [];
    if(typeof data.psnr_cover_container === "number") a.push(`PSNR(Cover→Container)：${data.psnr_cover_container} dB`);
    if(typeof data.psnr_secret_revealed === "number") a.push(`PSNR(Secret→Reveal)：${data.psnr_secret_revealed} dB`);
    if (data.norm_type) a.push(`Norm：${data.norm_type}`);
    mWrap.textContent = a.join(" ｜ ");
    mWrap.style.display = a.length ? "" : "none";
  }

  if(typeof bindZoomToNewImages==="function") bindZoomToNewImages();
}

/* 顯示 best_meta 與 edge_clean（若頁面放了對應元素就會顯示；沒有就忽略） */
function updateTileMetaUI(data){
  try{
    const meta = data?.best_meta;
    const edge = data?.edge_clean;
    const pick4 = data?.edge_clean_pick4;

    if($("wmTileMeta")){
      if(meta){
        const b = meta.box || [0,0,0,0];
        const metrics = meta.metrics || {};
        $("wmTileMeta").textContent =
          `最佳小圖：grid=${meta.grid}，box=[${b.join(", ")}]，score=${meta.score}` +
          (metrics && typeof metrics.zncc==="number"
            ? `（zncc=${metrics.zncc}｜grad=${metrics.zncc_grad}｜snr=${metrics.snr_n}｜g=${metrics.grad_n}｜bed=${metrics.border_edge_density}）`
            : "");
        $("wmTileMeta").style.display = "";
      }else{
        $("wmTileMeta").textContent = ""; $("wmTileMeta").style.display = "none";
      }
    }

    if($("wmEdgeClean")){
      if(edge?.edges_sorted){
        $("wmEdgeClean").textContent =
          `外圈由淨到髒（1..N）：${edge.edges_sorted.join(", ")}`;
        $("wmEdgeClean").style.display = "";
      }else{
        $("wmEdgeClean").textContent = ""; $("wmEdgeClean").style.display = "none";
      }
    }

    if($("wmEdgePick4")){
      if(Array.isArray(pick4) && pick4.length){
        $("wmEdgePick4").textContent = `建議四格（邊 / 角落）：${pick4.join(", ")}`;
        $("wmEdgePick4").style.display = "";
      }else{
        $("wmEdgePick4").textContent = ""; $("wmEdgePick4").style.display = "none";
      }
    }
  }catch(_){}
}

/* 報告（如果有就連結；沒有就清空） */
function updateWMReportUI(data){
  const report = data.report || (data.files?.report ? absoluteWMUrl(data.files.report) : null);
  const sess   = data.session_dir;

  let linked = false;
  if(report && /^https?:\/\//i.test(report)){
    bindOpenInNewTab("wmReportLink", report, "開啟案件報告");
    linked = true;
  }else if(report && report.startsWith("/")){
    bindOpenInNewTab("wmReportLink", `${WM_API_BASE}${report}`, "開啟案件報告");
    linked = true;
  }

  if(!linked && report){
    setText("wmReportPath", report);
    const p = $("wmReportPath");
    if(p){
      p.style.cursor = "copy";
      p.title = "點擊複製路徑";
      p.onclick = async ()=>{
        await copyToClipboard(report);
        p.classList.add("copied");
        setTimeout(()=>p.classList.remove("copied"), 600);
      };
    }
  }else{
    setText("wmReportPath",""); setLink("wmReportLink", null, null);
  }

  setText("wmZipPath",""); setLink("wmZipLink", null, null);
  if(sess){ setText("wmSessionDir", sess); } else { setText("wmSessionDir",""); }
}

/* =========================================================
   Best-tile（前端擷取，支援 4/3/2，並可優先遵循使用者的格數提示）
   後端缺少時才用
   ========================================================= */
async function _loadImage(url){
  const res = await fetch(url, {mode:"cors"});
  if(!res.ok) throw new Error(`HTTP ${res.status}`);
  const blob = await res.blob();
  return await createImageBitmap(blob);
}
function _toCanvas(img, sx=0, sy=0, sw=img.width, sh=img.height, outW=sw, outH=sh){
  const c = document.createElement("canvas");
  c.width = outW; c.height = outH;
  const ctx = c.getContext("2d", { willReadFrequently:true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, outW, outH);
  return c;
}
function _scoreTile(imageData, x, y, w, h, gxIdx=0, gyIdx=0, grid=1){
  const { data, width } = imageData;
  let sum=0, sum2=0, edges=0, borderEdges=0;
  const n=w*h;

  let t = Math.max(4, Math.floor(Math.min(w, h) * 0.06));
  if (t*2 >= Math.min(w,h)) t = Math.max(2, Math.floor(Math.min(w,h)/4));

  const isBorder = (i,j)=> (i < t || j < t || i >= w - t || j >= h - t);

  for(let j=0;j<h;j++){
    for(let i=0;i<w;i++){
      const px = ((y+j)*width + (x+i)) * 4;
      const r=data[px], g=data[px+1], b=data[px+2];
      const Y = 0.2126*r + 0.7152*g + 0.0722*b;
      sum += Y; sum2 += Y*Y;

      const nx = (i+1<w) ? ((y+j)*width + (x+i+1))*4 : px;
      const ny = (j+1<h) ? ((y+j+1)*width + (x+i))*4 : px;
      const Yr = 0.2126*data[nx] + 0.7152*data[nx+1] + 0.0722*data[nx+2];
      const Yd = 0.2126*data[ny] + 0.7152*data[ny+1] + 0.0722*data[ny+2];

      const e = (Math.abs(Y - Yr) > 8 || Math.abs(Y - Yd) > 8) ? 1 : 0;
      edges += e;
      if (e && isBorder(i,j)) borderEdges += 1;
    }
  }
  const mean = sum / n;
  const varY = Math.max(0, (sum2 / n) - mean*mean);
  const stdY = Math.sqrt(varY);
  const edgeDensity = edges / n;

  const borderPixels = Math.max(1, (2*t*w + 2*t*h - 4*t*t));
  const borderEdgeDensity = borderEdges / borderPixels;

  const atLeft   = (gxIdx === 0), atRight = (gxIdx === grid-1);
  const atTop    = (gyIdx === 0), atBottom= (gyIdx === grid-1);
  const cornerBias = ( (atLeft||atRight) && (atTop||atBottom) ) ? 0.03
                     : (atLeft||atRight||atTop||atBottom) ? 0.01 : 0.0;

  const extremePenalty = (mean < 6 || mean > 249) ? 0.25 : 0.0;

  return 0.55*(stdY/64) + 0.45*edgeDensity - 0.80*borderEdgeDensity + cornerBias - extremePenalty;
}
function _bestTileFromReveal(bitmap, hintGrid=null){
  const W = bitmap.width, H = bitmap.height;
  const full = _toCanvas(bitmap);
  const ctx = full.getContext("2d", { willReadFrequently:true });
  const imgData = ctx.getImageData(0,0,W,H);

  let best = {score:-1, x:0,y:0,w:W,h:H, grid:1};
  const base = [4,3,2];
  const grids = (hintGrid && [2,3,4].includes(hintGrid))
    ? [hintGrid, ...base.filter(g=>g!==hintGrid)]
    : base;

  for(const g of grids){
    const tw = Math.floor(W/g), th = Math.floor(H/g);
    if(tw<16 || th<16) continue;
    for(let gy=0; gy<g; gy++){
      for(let gx=0; gx<g; gx++){
        const x = gx*tw, y = gy*th, w = (gx===g-1) ? (W - x) : tw, h = (gy===g-1) ? (H - y) : th;
        const s = _scoreTile(imgData, x, y, w, h, gx, gy, g);
        if(s > best.score){ best = {score:s, x,y,w,h, grid:g}; }
      }
    }
    if(best.score >= 0.22) break;
  }
  return best;
}
async function extractBestRevealTileURL(revealUrl, upscaleTo=768, hintGrid=null){
  const bmp = await _loadImage(revealUrl);
  const best = _bestTileFromReveal(bmp, hintGrid);
  const scale = Math.max(best.w, best.h) > 0 ? upscaleTo / Math.max(best.w, best.h) : 1;
  const outW = Math.max(1, Math.round(best.w*scale));
  const outH = Math.max(1, Math.round(best.h*scale));
  const out = _toCanvas(bmp, best.x, best.y, best.w, best.h, outW, outH);
  return {
    dataURL: out.toDataURL("image/png"),
    box: {x:best.x,y:best.y,w:best.w,h:best.h},
    grid: best.grid,
    score: Number(best.score.toFixed(4))
  };
}

/* =============== 使用者的 reveal「格數提示」讀取 =============== */
function _getRevealGridHint(){
  const viaId = $("revealGridHint")?.value;
  const viaRadio = document.querySelector('input[name="revealGridHint"]:checked')?.value;
  const raw = (viaRadio ?? viaId ?? "").trim();
  const n = parseInt(raw, 10);
  return [2,3,4].includes(n) ? n : null; // null = Auto
}

/* === 區塊鏈驗證卡片輔助 === */
function _bcCard(state, title, detail){
  const card  = $("bcVerifyCard");
  const tEl   = $("bcVerifyTitle");
  const dEl   = $("bcVerifyDetail");
  if(!card) return;
  card.className = "bc-verify-card " + state;   // bc-checking | bc-ok | bc-fail
  if(tEl) tEl.textContent = title;
  if(dEl) dEl.textContent = detail || "";
}
function _bcCardHide(){
  const card = $("bcVerifyCard");
  if(card){ card.className = "bc-verify-card"; card.style.display = "none"; }
}

/* ===================== 破譯真真（5001） ===================== */
async function runReveal(){
  const attack = $("attackFile")?.files?.[0];
  if(!attack){
    displayResponse("wmRevealResp","請選擇要解碼的圖片（container/attacked）。",true);
    return;
  }
  const refImg = $("secretRef")?.files?.[0] || $("referenceFile")?.files?.[0] || null;
  const preferNorm = _getPreferNorm();
  const gridHint = _getRevealGridHint();
  const secretCountHint = gridHint ? (gridHint*gridHint) : null;

  const btn = $("revealBtn");
  if(btn){ btn.disabled = true; btn.classList.add("disabled"); btn.setAttribute("aria-busy","true"); }

  showSpinner("wmRevealSpinner","處理中…");
  displayResponse("wmRevealResp","");
  _bcCard("bc-checking", "區塊鏈驗證中…", "正在確認此圖片是否已在區塊鏈中登記");

  try{
    const fd = new FormData();
    fd.append("attack", attack);
    if(refImg) fd.append("secret_ref", refImg);
    if(preferNorm) fd.append("prefer_norm", preferNorm);
    if(gridHint){ fd.append("hint_grid", String(gridHint)); }
    if(secretCountHint){ fd.append("hint_secret_count", String(secretCountHint)); }
    if(gridHint){ fd.append("secret_grid_hint", String(gridHint)); fd.append("hint_secret_tiles", String(secretCountHint)); }
    _appendCollectionFields(fd);

    const resp = await fetch(`${WM_API_BASE}/external_reveal`, { method:"POST", body:fd, mode:"cors" });

    // 區塊鏈驗證失敗（403）
    if(resp.status === 403){
      let errData = {}; try{ errData = await resp.json(); }catch(_){}
      const detail = errData.detail || "";
      const sha = errData.image_sha256 ? `\nSHA-256：${errData.image_sha256}` : "";
      if(detail === "not_registered"){
        _bcCard("bc-fail",
          "區塊鏈驗證失敗：此圖片未登記",
          "此圖片的 SHA-256 雜湊值未在區塊鏈中找到記錄。\n請確認您上傳的是由本系統「施法加印」所產生的 container 圖片，而非原始或其他來源的圖片。" + sha
        );
      } else {
        _bcCard("bc-fail",
          "區塊鏈驗證失敗：鏈已被竄改",
          "區塊鏈完整性校驗未通過，資料可能遭到篡改。\n詳細原因：" + (detail || "unknown") + sha
        );
      }
      hideSpinner("wmRevealSpinner");
      return;
    }

    if(!resp.ok){
      let t=""; try{ t=await resp.text(); }catch(_){}
      const msg = `HTTP ${resp.status}${t?`: ${t.substring(0,200)}`:""}`;
      const hint = (location.protocol==="https:" && WM_API_BASE.startsWith("http://"))
        ? "無法連線 Watermark 後端（5001）。（HTTPS↔HTTP 混合內容被瀏覽器擋下）"
        : "無法連線 Watermark 後端（5001）。";
      _bcCardHide();
      displayResponse("wmRevealResp",`${hint} 詳細：${msg}`,true);
      return;
    }

    const data = await resp.json().catch(()=> ({}));
    const ok = (data?.ok || data?.status==="ok");

    if(!ok){
      const backendMsg = data?.error || "解碼失敗";
      const detail = data?.stderr || data?.stdout || "";
      const tail = detail ? `｜stderr: ${String(detail).slice(-200)}` : "";
      _bcCardHide();
      displayResponse("wmRevealResp", `Watermark 後端回報錯誤（5001）。${backendMsg}${tail}`, true);
      return;
    }

    // 區塊鏈驗證通過
    if(data.blockchain_verified && data.blockchain){
      const bc = data.blockchain;
      _bcCard("bc-ok",
        "區塊鏈驗證通過",
        `區塊索引：#${bc.block_index}　SHA-256：${(bc.image_sha256||"").slice(0,16)}…\n` +
        `區塊雜湊：${(bc.block_hash||"").slice(0,24)}…　原始嵌入工作：${bc.embed_job_id||""}`
      );
    } else {
      _bcCardHide();
    }

    // 正規化欄位
    if(data?.files?.reveal){
      data.reveal_dataurl = absoluteWMUrl(data.files.reveal);
      data.images = data.images || {};
      data.images.revealed = data.reveal_dataurl;
    }
    if(data.images?.revealed){
      data.images.revealed = pathToOpenImageURL(data.images.revealed);
    }
    if(data.images?.reveal){
      const u = pathToOpenImageURL(data.images.reveal);
      data.reveal_dataurl = u;
      data.images.revealed = u;
    }
    if(data.reveal && !data.reveal_dataurl){
      data.reveal_dataurl = pathToOpenImageURL(data.reveal);
      data.images = data.images || {};
      if(!data.images.revealed) data.images.revealed = data.reveal_dataurl;
    }
    if(data.images?.tile_best){
      data.images.tile_best = pathToOpenImageURL(data.images.tile_best);
    }
    if(data.images?.tile_best_x4){
      data.images.tile_best_x4 = pathToOpenImageURL(data.images.tile_best_x4);
    }
    if(data.images?.tile_best_256){
      data.images.tile_best_256 = pathToOpenImageURL(data.images.tile_best_256);
    }

    let bestUrl = data?.images?.tile_best_256 || pickBestRevealURL(data);
    let usedClientBest = false;

    if(!bestUrl){
      const revealUrl = data.reveal_dataurl || data.images?.revealed || data.images?.reveal || data.reveal || null;
      if(revealUrl){
        try{
          const best = await extractBestRevealTileURL(revealUrl, 768, gridHint || null);
          bestUrl = best.dataURL;
          usedClientBest = true;
          console.log("[client-best-tile]", best);
        }catch(e){
          bestUrl = revealUrl;
        }
      }
    }

    const finalUrl = bestUrl;
    if(!finalUrl){
      displayResponse("wmRevealResp","後端未提供可用的結果圖。", true);
      return;
    }

    const resultsWrap = $("imageResults2");
    const rImg = $("imgRevealOnly");
    const prevDup = $("originalImage2_dup");

    const fileReader = new FileReader();
    const ori2 = $("originalImage2");
    fileReader.onload = (e)=>{
      if(ori2){ ori2.src = e.target.result; ori2.style.display = "block"; ori2.classList.add("zoomable"); }
      if(prevDup){ prevDup.src = e.target.result; prevDup.style.display = "block"; prevDup.classList.add("zoomable"); }
      if(typeof bindZoomToNewImages==="function") bindZoomToNewImages();
    };
    fileReader.readAsDataURL(attack);

    if(rImg){
      rImg.src = finalUrl;
      rImg.style.display="block";
      rImg.classList.add("zoomable");
    }
    if(resultsWrap) resultsWrap.style.display="block";

    const stem = (attack.name||"reveal").replace(/\.[^.]+$/,"");
    bindDownloadLink("downloadBestReveal", finalUrl, `${stem}_revealed.png`);
    bindDownloadLink("downloadRevealed",   finalUrl, `${stem}_revealed.png`);
    bindDownloadLink("downloadReveal",     finalUrl, `${stem}_revealed.png`);
    bindDownloadLink("downloadDecoded",    finalUrl, `${stem}_revealed.png`);

    updateTileMetaUI(data);

    const hintEcho = gridHint ? `（優先依你選的 ${gridHint}×${gridHint}；secret≈${secretCountHint} 張）` : "";
    const msg = (data?.images?.tile_best_256)
      ? `完成 ✔️（最佳小圖 256×256）${hintEcho}`
      : (usedClientBest ? `完成 ✔️（已自動挑選最完整小圖）${hintEcho}` :
         ((data.images?.tile_best || data.images?.tile_best_x4) ? `完成 ✔️（最佳小圖）${hintEcho}` : `完成 ✔️${hintEcho}`));
    displayResponse("wmRevealResp", msg);

    if(typeof bindZoomToNewImages==="function") bindZoomToNewImages();
  }catch(err){
    const message = String(err?.message || err) || "";
    const hint = (location.protocol==="https:" && WM_API_BASE.startsWith("http://"))
      ? "無法連線 Watermark 後端（5001）。（HTTPS↔HTTP 混合內容被瀏覽器擋下）"
      : "無法連線 Watermark 後端（5001）。";
    displayResponse("wmRevealResp",`${hint} 詳細：${message}`,true);
  }finally{
    hideSpinner("wmRevealSpinner");
    const btn = $("revealBtn");
    if(btn){ btn.disabled = false; btn.classList.remove("disabled"); btn.setAttribute("aria-busy","false"); }
  }
}

/* ===================== 圖片放大（共用｜全站通用美化版） ===================== */
let __zoom_inited = false;
let __zoom_state = { scale: 1 };

function initZoomModal(){
  if(__zoom_inited) return;
  __zoom_inited = true;

  const modal   = $("imageModal");
  const img     = $("modalImage");
  const btnIn   = $("wmZoomIn");
  const btnOut  = $("wmZoomOut");
  const btnRst  = $("wmZoomReset");

  if(!modal || !img) return;

  function applyZoom(){ img.style.transform = `scale(${__zoom_state.scale})`; }

  function openModal(src){
    if(src) img.src = src;
    __zoom_state.scale = 1;
    applyZoom();
    modal.setAttribute("aria-hidden","false");
    modal.style.display = "block";
    document.documentElement.style.overflow = "hidden";
  }
  function closeModal(){
    modal.setAttribute("aria-hidden","true");
    modal.style.display = "none";
    document.documentElement.style.overflow = "";
  }
  function zoomIn(){ __zoom_state.scale = Math.min(4, __zoom_state.scale + 0.1); applyZoom(); }
  function zoomOut(){ __zoom_state.scale = Math.max(0.2, __zoom_state.scale - 0.1); applyZoom(); }
  function resetZoom(){ __zoom_state.scale = 1; applyZoom(); }

  window.closeModal = closeModal;
  window.zoomIn     = zoomIn;
  window.zoomOut    = zoomOut;
  window.resetZoom  = resetZoom;

  modal.addEventListener("click",(e)=>{
    if(e.target && (e.target.matches("[data-close]") || e.target.closest("[data-close]"))){
      closeModal();
    }
  });

  document.addEventListener("keydown",(e)=>{
    if(e.key === "Escape" && modal.getAttribute("aria-hidden") === "false"){
      closeModal();
    }
  });

  if(btnIn)  btnIn.addEventListener("click", zoomIn);
  if(btnOut) btnOut.addEventListener("click", zoomOut);
  if(btnRst) btnRst.addEventListener("click", resetZoom);

  initZoomModal.open = openModal;
}

function bindZoomToNewImages(root=document){
  initZoomModal();
  const modal = $("imageModal");
  const modalImg = $("modalImage");
  if(!modal || !modalImg || !initZoomModal.open) return;

  root.querySelectorAll(".zoomable").forEach((img)=>{
    if(img.__zoomBound) return;
    img.__zoomBound = true;
    img.addEventListener("click",(e)=>{
      e.preventDefault();
      const src = img.src || img.getAttribute("src");
      if(src) initZoomModal.open(src);
    });
  });
}

/* ===================== 首頁輪播 / FX ===================== */
function initSlider(rootId="topSlider", intervalMs=8000){
  const root=$(rootId); if(!root) return;
  const track=root.querySelector(".slider-track");
  const slides=Array.from(root.querySelectorAll(".slide"));
  const btnPrev=root.querySelector(".slider-arrow.prev");
  const btnNext=root.querySelector(".slider-arrow.next");
  const dotsWrap=root.querySelector(".slider-dots");
  if(!track||slides.length===0||!dotsWrap) return;

  let idx=0, timer=null, isHover=false, touchStartX=0, touchDeltaX=0;
  dotsWrap.innerHTML="";
  slides.forEach((_,i)=>{
    const b=document.createElement("button");
    b.type="button"; b.className="dot"+(i===0?" active":"");
    b.setAttribute("aria-label",`第 ${i+1} 張`);
    b.addEventListener("click",()=>goTo(i,true));
    dotsWrap.appendChild(b);
  });
  const dots=Array.from(dotsWrap.querySelectorAll(".dot"));

  function replayRevealWithin(slide){
    const kids = slide.querySelectorAll(".reveal");
    kids.forEach((el,i)=>{
      el.classList.remove("visible");
      void el.offsetWidth;
      el.style.setProperty("--d", `${i*120}ms`);
      el.classList.add("visible");
    });
  }

  function update(){
    track.style.transform=`translateX(${-idx*100}%)`;
    dots.forEach((d,i)=>d.classList.toggle("active", i===idx));
    replayRevealWithin(slides[idx]);
    root.dispatchEvent(new CustomEvent("slidechange",{detail:{index:idx}}));
  }
  function goTo(i,user=false){ idx=(i+slides.length)%slides.length; update(); if(user) restartAuto(); }
  function next(user=false){ goTo(idx+1,user); }
  function prev(user=false){ goTo(idx-1,user); }
  function startAuto(){ stopAuto(); timer=setInterval(()=>{ if(!isHover && document.visibilityState==="visible") next(false); }, intervalMs); }
  function stopAuto(){ if(timer){ clearInterval(timer); timer=null; } }
  function restartAuto(){ stopAuto(); startAuto(); }

  if(btnPrev) btnPrev.addEventListener("click",()=>prev(true));
  if(btnNext) btnNext.addEventListener("click",()=>next(true));
  root.addEventListener("mouseenter",()=>{ isHover=true; });
  root.addEventListener("mouseleave",()=>{ isHover=false; });
  root.addEventListener("touchstart",(e)=>{ touchStartX=e.touches[0].clientX; touchDeltaX=0; stopAuto(); },{passive:true});
  root.addEventListener("touchmove",(e)=>{ touchDeltaX=e.touches[0].clientX - touchStartX; },{passive:true});
  root.addEventListener("touchend",()=>{ const TH=40; if(Math.abs(touchDeltaX)>TH){ if(touchDeltaX<0) next(true); else prev(true); } startAuto(); });
  root.setAttribute("tabindex","0");
  root.addEventListener("keydown",(e)=>{ if(e.key==="ArrowRight") next(true); else if(e.key==="ArrowLeft") prev(true); });
  document.addEventListener("visibilitychange",()=>{ if(document.visibilityState==="visible") startAuto(); else stopAuto(); });

  update(); startAuto();
}
function initRevealAndParallax(){
  if (window.__fxInited) return;
  window.__fxInited = true;
  stripRevealInNav();
  document.documentElement.classList.add("fx-ready");

  const all = Array.from(document.querySelectorAll(".reveal, [data-stagger]"))
    .filter(el => !el.closest('nav'));

  const io = new IntersectionObserver((entries)=>{
    entries.forEach((en)=>{
      if (!en.isIntersecting) return;
      const el = en.target;
      if (el.dataset && el.dataset.stagger){
        const ms = parseInt(el.dataset.stagger,10) || 160;
        el.querySelectorAll(".reveal").forEach((child,i)=>{
          child.style.setProperty("--d", `${i*ms}ms`);
          child.classList.add("visible");
        });
      }else{
        el.classList.add("visible");
      }
      io.unobserve(el);
    });
  }, { threshold: 0.15 });
  all.forEach(el=> io.observe(el));

  const items = Array.from(document.querySelectorAll("[data-parallax]"));
  const onScroll = ()=>{
    const top = window.scrollY || 0;
    items.forEach(el=>{
      const k = parseFloat(el.dataset.parallax || "6");
      const rect = el.getBoundingClientRect();
      const y = ((top + rect.top) / k) * 0.10;
      el.style.setProperty("--py", `${y.toFixed(2)}px`);
    });
  };
  onScroll();
  window.addEventListener("scroll", onScroll, { passive:true });
  window.addEventListener("resize", onScroll, { passive:true });

  console.log("[FX] reveal/parallax ready");
}
function initLoopFX(){
  const maps = {
    up:    [{transform:'translateY(0)'},{transform:'translateY(-8px)'}],
    down:  [{transform:'translateY(0)'},{transform:'translateY(8px)'}],
    left:  [{transform:'translateX(0)'},{transform:'translateX(-10px)'}],
    right: [{transform:'translateX(0)'},{transform:'translateX(10px)'}],
    pop:   [{transform:'scale(1)'},{transform:'scale(1.06)'}],
  };
  document.querySelectorAll("[data-loop]").forEach(el=>{
    const t = (el.dataset.loop || "up").toLowerCase();
    const dur = parseInt(el.dataset.loopDur || "1600", 10);
    const easing = "ease-in-out";
    const frames = maps[t] || maps.up;
    el.animate(frames, {duration: dur, iterations: Infinity, direction: "alternate", easing});
  });
}

/* ===================== DOM Ready ===================== */
document.addEventListener("DOMContentLoaded", ()=>{
  stripRevealInNav();

  /* ---- 施法加印 ---- */
  const embedBtn=$("embedBtn");
  if(embedBtn){ embedBtn.addEventListener("click", ()=>runEmbedGeneric("coverFile","secretFile")); }
  const singleBtn=$("singleEmbedBtn");
  if(singleBtn){ singleBtn.addEventListener("click", ()=>runEmbedGeneric("singleCover", null)); }

  /* ---- 破譯真身：選檔只預覽；按執行才顯示結果＋提示 ---- */
  const attackFile = $("attackFile");
  const revealBtn  = $("revealBtn");
  const prev1 = $("originalImage2");
  const prev2 = $("originalImage2_dup");
  const resultsWrap2 = $("imageResults2");
  if(resultsWrap2) resultsWrap2.style.display = "none";

  const updateRevealBtnState = ()=>{
    const has = !!(attackFile && attackFile.files && attackFile.files[0]);
    if(revealBtn){
      revealBtn.disabled = !has;
      revealBtn.classList.toggle("disabled", !has);
      revealBtn.setAttribute("aria-busy","false");
    }
  };
  updateRevealBtnState();

  if(attackFile){
    attackFile.addEventListener("change", (e)=>{
      updateRevealBtnState();
      const f = e.target.files && e.target.files[0];

      if(resultsWrap2) resultsWrap2.style.display = "none";
      const rImg = $("imgRevealOnly"); if(rImg){ rImg.src=""; rImg.style.display="none"; }
      displayResponse("wmRevealResp",""); hideSpinner("wmRevealSpinner");

      if(!f || (f.type && !f.type.startsWith("image/"))){
        if(prev1){ prev1.style.display="none"; prev1.src=""; }
        if(prev2){ prev2.style.display="none"; prev2.src=""; }
        return;
      }
      const fr = new FileReader();
      fr.onload = ev=>{
        const url = ev.target.result;
        [prev1, prev2].forEach(img=>{
          if(!img) return;
          img.src = url; img.style.display = "block"; img.classList.add("zoomable");
        });
        if(typeof bindZoomToNewImages==="function") bindZoomToNewImages(document);
      };
      fr.readAsDataURL(f);
    });
  }
  if(revealBtn){
    revealBtn.addEventListener("click", (e)=>{ e.preventDefault(); revealBtn.setAttribute("aria-busy","true"); runReveal(); });
  }

  // 健康檢查鉤子
  window.__wmPing  = async()=>{
    try{
      const r=await fetch(`${WM_API_BASE}/health`);
      console.info("[5001]/health →", r.status);
    }catch(e){
      console.warn("[5001]/health error:", e);
    }
  };
  // 首頁輪播 / 動效
  initSlider("topSlider", 8000);
  initRevealAndParallax();
  initLoopFX();

  // 初始化放大視窗並綁定現有圖片
  initZoomModal();
  bindZoomToNewImages(document);

  console.log("[main] ready.", { API_BASE, WM_API_BASE, ORIGIN_BASE });
});

/* ===== 導覽列透明/實色切換 + 回頂按鈕顯示 ===== */
(function(){
  const nav = document.querySelector('nav');
  const backTop = document.getElementById('backToTop');

  function onScroll(){
    if(nav){
      if (window.scrollY > 8) nav.classList.add('nav-solid');
      else nav.classList.remove('nav-solid');
    }
    if(backTop){
      const show = window.scrollY > 300;
      backTop.style.opacity = show ? '1' : '0';
      backTop.style.pointerEvents = show ? 'auto' : 'none';
    }
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  if(backTop){
    backTop.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }
})();
