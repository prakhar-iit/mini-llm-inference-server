(function(){
const store={get(k){try{return localStorage.getItem("iec:"+k)}catch(e){return null}},set(k,v){try{localStorage.setItem("iec:"+k,v)}catch(e){}}};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const mods=ITEMS.filter(i=>i.kind==="module");
const unitOf=k=>UNITS.find(u=>u.id===k);
const isDone=m=>store.get(m.id+"-lesson")==="1"&&store.get(m.id+"-lab")==="1";
const view=document.getElementById("view");

function rail(active){
  const ul=document.getElementById("unitList");
  ul.innerHTML=UNITS.map(u=>{
    const ms=mods.filter(m=>m.unit===u.id);
    const cnt=ms.length?`${ms.filter(isDone).length}/${ms.length}`:"";
    const cur=active===u.key;
    const sub=cur&&ms.length?`<ul class="sublist">${ms.map(m=>`<li><a href="#${m.id}" class="${isDone(m)?"done":""}">${m.n}. ${esc(m.title)}</a></li>`).join("")}</ul>`:"";
    return `<li><a href="#${u.key}" ${cur?'aria-current="page"':""}><span class="uid">${u.id==="A"?"ref":"U"+u.id}</span><span>${esc(u.title)}</span><span class="ucount">${cnt}</span></a>${sub}</li>`;
  }).join("");
  const d=mods.filter(isDone).length;
  document.getElementById("progBar").style.width=(100*d/mods.length)+"%";
  document.getElementById("progTxt").textContent=`${d} of ${mods.length} modules done`;
}

function tagHtml(t){if(Array.isArray(t))return `<span class="tag ${t[1]}">${esc(t[0])}-bound</span>`;return `<span class="tag">${esc(t)}</span>`}
const pill={ready:'<span class="pill ready">Script ready · pending EC2 run</span>',planned:'<span class="pill planned">Planned</span>',optional:'<span class="pill optional">Optional</span>'};

function moduleHtml(m){
  const d=m.diag&&DIAG[m.diag];
  const L=m.lab;
  return `<article class="module" id="${m.id}">
  <div class="mhead"><span class="mnum">Module ${m.n}</span><h2>${esc(m.title)}</h2></div>
  <div class="tags">${m.tags.map(tagHtml).join("")}</div>
  <div class="part">Lesson</div>
  <div class="lesson">
    <div class="obj"><b>You will be able to</b><ul>${m.obj.map(o=>`<li>${esc(o)}</li>`).join("")}</ul></div>
    ${m.lesson}
    <div class="formula">${esc(m.formula)}</div>
    ${m.example?`<div class="example">${m.example}</div>`:""}
  </div>
  ${d?`<figure>${d.svg}<figcaption>${esc(d.cap)}</figcaption></figure>`:""}
  <div class="part">Predict before you run</div>
  <div class="predict">${m.predict.map((q,i)=>`<div class="q"><span class="qn">P${i+1}</span><p>${esc(q)}</p><textarea id="${m.id}-q${i}" aria-label="Your prediction ${i+1}" placeholder="A number and a reason">${esc(store.get(m.id+"-q"+i)||"")}</textarea></div>`).join("")}</div>
  <div class="part">Lab on EC2</div>
  <section class="lab" aria-label="Lab">
    <div class="labtop"><h3>${esc(L.name)}</h3>${pill[m.status]}</div>
    <p class="why"><b>Why we run it.</b> ${esc(L.why)}</p>
    <div><p class="labh">Setup</p><div class="spec">${L.spec.map(([k,v])=>`<div>${esc(k)}</div><div>${v}</div>`).join("")}</div></div>
    <div><p class="labh">Metrics to watch</p><div class="metrics">${L.metrics.map(([k,v])=>`<div class="metric"><b>${esc(k)}</b>${v?`<span>${esc(v)}</span>`:""}</div>`).join("")}</div></div>
    <div class="expect"><b>Expected result</b>${esc(L.expected)}</div>
    <div class="measured">Measured result: ${m.measured?esc(m.measured):m.status==="ready"?"pending EC2 run. Record it in docs/studies/ with the gap explained.":"pending. The milestone is not built yet."}</div>
  </section>
  <div class="part">Takeaway</div>
  <div class="take">
    <div><h4>What it teaches</h4><p style="margin:0">${esc(m.learn)}</p><p class="nextp" style="margin:.6em 0 0"><b>Leads to:</b> ${esc(m.next)}</p></div>
    <div><h4>Check yourself</h4><ul>${m.check.map(c=>`<li>${esc(c)}</li>`).join("")}</ul></div>
  </div>
  <div class="checks">
    <label><input type="checkbox" id="${m.id}-lesson" ${store.get(m.id+"-lesson")==="1"?"checked":""}> Lesson read</label>
    <label><input type="checkbox" id="${m.id}-lab" ${store.get(m.id+"-lab")==="1"?"checked":""}> Lab done</label>
  </div>
  </article>`;
}
function decisionHtml(d){
  const open=d.status==="open";
  return `<aside class="decision ${open?"open":""}"><div class="dtop"><span class="flag">${open?"⚑ Open decision":"✓ Decided"} · ${esc(d.n)}</span><h3>${esc(d.title)}</h3></div>
  <p>${esc(d.choice)}</p><p class="rej"><b>Reason:</b> ${esc(d.why)}</p>${d.rej?`<p class="rej">${esc(d.rej)}</p>`:""}</aside>`;
}
function unitHead(u){
  return `<header class="unit-head"><span class="eyebrow">${u.id==="A"?"Reference":"Unit "+u.id}</span><h1>${esc(u.title)}</h1>
  <p><b>Goal.</b> ${esc(u.goal)}</p>${u.outcome?`<p><b>Outcome.</b> ${esc(u.outcome)}</p>`:""}
  ${u.facts?`<div class="facts">${u.facts.map(([k,v])=>`<div class="fact"><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join("")}</div>`:""}</header>`;
}
function pager(u){
  const i=UNITS.indexOf(u),p=UNITS[i-1],n=UNITS[i+1];
  return `<nav class="pager">${p?`<a href="#${p.key}">← ${esc(p.title)}</a>`:'<a href="#home">← Course map</a>'}${n?`<a href="#${n.key}">${esc(n.title)} →</a>`:"<span></span>"}</nav>`;
}
function home(){
  const studies=mods.length;
  return `<header class="unit-head"><span class="eyebrow">Course map</span><h1>Inference Engineering</h1>
  <p>A course built from your own roadmap. Each module teaches a concept, asks you to predict a number from it, then runs an experiment on EC2 that tests the prediction. Units follow the build order, from a bare generation loop to Kimi K2 on two nodes.</p>
  <div class="facts"><div class="fact"><b>Units</b><span>7 plus reference</span></div><div class="fact"><b>Modules</b><span>${studies}, each lesson → lab</span></div><div class="fact"><b>Built so far</b><span>M0–M6 (tags m0–m6)</span></div><div class="fact"><b>Measured so far</b><span>Nothing yet: all runs pending on EC2</span></div></div></header>
  <div class="map">${UNITS.filter(u=>u.id!=="A").map(u=>{const ms=mods.filter(m=>m.unit===u.id);return `<a class="mapu" href="#${u.key}"><span class="k">${u.id}</span><div><h3>${esc(u.title)}</h3><p>${esc(u.goal)}</p><div class="chips">${ms.map(m=>`<span class="chip">${esc(m.lab.name.split(" · ")[0])}</span>`).join("")}</div></div></a>`}).join("")}
  <a class="mapu" href="#unit-appendix"><span class="k">ref</span><div><h3>Reference</h3><p>Formula sheet, GPU and link numbers, decision log, glossary, kernel track.</p></div></a></div>
  <div class="sec"><h2>MVP cut</h2><p>If time runs short, the core you need for interviews is Unit 1a, chunked prefill (module 9), routing (16) and P/D disaggregation (17). Reliability, multi-node N1–N4, P2 and P4 are cut first. Unit 5 is a capstone; applications never wait for it.</p></div>`;
}
function render(){
  let h=(location.hash||"#home").slice(1);
  let u=UNITS.find(x=>x.key===h), target=null;
  const m=mods.find(x=>x.id===h); if(m){u=unitOf(m.unit);target=m.id}
  if(!u&&h!=="home")h="home";
  if(!u){view.innerHTML=home();rail("home");window.scrollTo(0,0);return}
  let body=unitHead(u);
  if(u.id==="0")body+=ORIENTATION;
  if(u.id==="A")body+=APPENDIX;
  body+=ITEMS.filter(i=>i.unit===u.id).map(i=>i.kind==="module"?moduleHtml(i):decisionHtml(i)).join("");
  body+=pager(u);
  view.innerHTML=body; rail(u.key);
  document.getElementById("rail").classList.remove("openr");
  document.getElementById("menuBtn").setAttribute("aria-expanded","false");
  if(target){const el=document.getElementById(target);if(el)requestAnimationFrame(()=>el.scrollIntoView())}else window.scrollTo(0,0);
}
view.addEventListener("change",e=>{const t=e.target;if(t.type==="checkbox"){store.set(t.id,t.checked?"1":"0");const h=(location.hash||"").slice(1);const m=mods.find(x=>x.id===h);rail(m?unitOf(m.unit).key:h)}});
view.addEventListener("input",e=>{const t=e.target;if(t.tagName==="TEXTAREA")store.set(t.id,t.value)});
document.getElementById("menuBtn").addEventListener("click",()=>{const r=document.getElementById("rail");const o=r.classList.toggle("openr");document.getElementById("menuBtn").setAttribute("aria-expanded",o)});
window.addEventListener("hashchange",render);
render();
})();
