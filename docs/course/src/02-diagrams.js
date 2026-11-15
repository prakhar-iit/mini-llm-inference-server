const DIAG = {
roofline:{cap:"Roofline. Left of the balance point the GPU waits on memory; right of it, on arithmetic. Decode at batch 1 sits far left; a long prefill sits on the flat roof.",svg:`<svg viewBox="0 0 640 250" role="img" aria-label="Roofline diagram">
<rect x="60" y="30" width="270" height="170" class="s-ms"/><rect x="330" y="30" width="270" height="170" class="s-cs"/>
<line x1="60" y1="200" x2="600" y2="200" class="s-axis" stroke-width="1.5"/><line x1="60" y1="200" x2="60" y2="25" class="s-axis" stroke-width="1.5"/>
<polyline points="60,190 330,55 600,55" fill="none" class="s-ast" stroke-width="3"/>
<line x1="330" y1="55" x2="330" y2="200" class="s-axis" stroke-dasharray="4 4"/>
<circle cx="88" cy="176" r="6" class="s-m"/><text x="100" y="172" class="s-ink">decode, batch 1 (~1 FLOP/byte)</text>
<circle cx="250" cy="95" r="6" class="s-m"/><text x="140" y="118" class="s-ink">decode, batch 64</text>
<circle cx="520" cy="55" r="6" class="s-c"/><text x="455" y="45" class="s-ink">prefill 4K tokens</text>
<text x="70" y="45" class="s-m t-lg">memory-bound</text><text x="480" y="190" class="s-c t-lg">compute-bound</text>
<text x="338" y="190" class="s-mut">balance point ≈ peak FLOP/s ÷ bandwidth</text>
<text x="300" y="225" class="s-mut">arithmetic intensity (FLOPs per byte read), log scale →</text>
<text x="18" y="120" class="s-mut" transform="rotate(-90 18 120)">achieved FLOP/s →</text>
</svg>`},
prefilldecode:{cap:"One request over time. Prefill pushes every prompt token through the model in one pass; decode then makes one pass per output token, each re-reading all the weights.",svg:`<svg viewBox="0 0 640 170" role="img" aria-label="Prefill then decode timeline">
<line x1="40" y1="120" x2="610" y2="120" class="s-axis"/>
<rect x="40" y="50" width="170" height="60" rx="4" class="s-c"/><text x="52" y="85" class="t-lg" fill="#fff">prefill: 4,096 tokens</text>
<g class="s-m">${Array.from({length:22},(_,i)=>`<rect x="${218+i*17.5}" y="70" width="13" height="40" rx="2"/>`).join("")}</g>
<text x="300" y="60" class="s-ink">decode: 1 token per step</text>
<line x1="210" y1="130" x2="210" y2="142" class="s-axis"/><text x="150" y="158" class="s-mut">TTFT ends here</text>
<text x="420" y="145" class="s-mut">gap between bars = ITL / TPOT</text>
<text x="40" y="30" class="s-mut">compute-bound: many tokens share one weight read</text><text x="360" y="30" class="s-mut">memory-bound: weights read per token</text>
</svg>`},
queue:{cap:"Mean wait grows as ρ/(1−ρ). Fine at half load, explosive near 1. ρ = arrival rate × service time.",svg:`<svg viewBox="0 0 640 240" role="img" aria-label="Queueing delay curve">
<line x1="60" y1="200" x2="600" y2="200" class="s-axis"/><line x1="60" y1="200" x2="60" y2="20" class="s-axis"/>
<polyline fill="none" class="s-ast" stroke-width="3" points="${Array.from({length:46},(_,i)=>{const r=i/50;const w=Math.min(r/(2*(1-r)),9);return `${60+r*540},${200-w*19}`}).join(" ")}"/>
<line x1="60" y1="120" x2="600" y2="120" class="s-cst" stroke-dasharray="5 4"/><text x="70" y="113" class="s-c">p99 TTFT SLO</text>
<line x1="546" y1="200" x2="546" y2="30" class="s-axis" stroke-dasharray="3 4"/><text x="552" y="45" class="s-mut">ρ = 0.9</text>
<text x="60" y="218" class="s-mut">0</text><text x="324" y="218" class="s-mut">0.5</text><text x="592" y="218" class="s-mut">1</text>
<text x="250" y="236" class="s-mut">utilisation ρ →</text><text x="18" y="130" class="s-mut" transform="rotate(-90 18 130)">queue wait →</text>
</svg>`},
batching:{cap:"Static batching holds every slot until the longest request finishes (hatched = wasted rows). Continuous batching refills a slot the step after it frees.",svg:`<svg viewBox="0 0 640 260" role="img" aria-label="Static vs continuous batching">
<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" class="s-axis" stroke-width="2"/></pattern></defs>
<text x="20" y="22" class="t-lg s-ink">Static (M3)</text>
${[[200],[90],[140],[60]].map((r,i)=>`<rect x="110" y="${32+i*24}" width="${r[0]}" height="18" class="s-m"/><rect x="${110+r[0]}" y="${32+i*24}" width="${200-r[0]}" height="18" fill="url(#hatch)"/>`).join("")}
${[[120],[170],[80],[110]].map((r,i)=>`<rect x="314" y="${32+i*24}" width="${r[0]}" height="18" class="s-a"/><rect x="${314+r[0]}" y="${32+i*24}" width="${170-r[0]}" height="18" fill="url(#hatch)"/>`).join("")}
<text x="20" y="150" class="t-lg s-ink">Continuous (M4)</text>
${[[[200,"m"]],[[90,"m"],[120,"a"]],[[140,"m"],[170,"a"]],[[60,"m"],[80,"a"],[110,"m"]]].map((row,i)=>{let x=110;return row.map(([w,c])=>{const s=`<rect x="${x}" y="${160+i*24}" width="${w}" height="18" class="s-${c}"/>`;x+=w+4;return s}).join("")}).join("")}
${[0,1,2,3].map(i=>`<text x="40" y="${46+i*24}" class="s-mut">slot ${i}</text><text x="40" y="${174+i*24}" class="s-mut">slot ${i}</text>`).join("")}
<text x="500" y="200" class="s-mut">same work,</text><text x="500" y="214" class="s-mut">finishes sooner</text>
</svg>`},
paged:{cap:"Contiguous slots reserve max_model_len per request, so most of each slot is empty. Paged KV hands out 16-token blocks on demand; only the last block of each request is partly empty.",svg:`<svg viewBox="0 0 640 250" role="img" aria-label="Contiguous versus paged KV memory">
<text x="20" y="22" class="t-lg s-ink">Contiguous slots (M4)</text>
${[0.18,0.5,0.1].map((f,i)=>`<rect x="20" y="${34+i*26}" width="280" height="20" class="s-p"/><rect x="20" y="${34+i*26}" width="${280*f}" height="20" class="s-m"/>`).join("")}
<text x="20" y="128" class="s-mut">3 requests fit; most reserved memory is empty</text>
<text x="340" y="22" class="t-lg s-ink">Paged pool (M5)</text>
${Array.from({length:60},(_,i)=>{const c=["m","a","c","m","a","c","o"][i%7];const used=i<52;return `<rect x="${340+(i%12)*23}" y="${34+Math.floor(i/12)*22}" width="19" height="18" rx="2" class="${used?"s-"+c:"s-p"}"/>`}).join("")}
<text x="340" y="160" class="s-mut">7 requests share one pool; colour = owner</text>
<text x="20" y="190" class="s-ink">block table, request A:</text>
${[12,3,40,7].map((b,i)=>`<rect x="${190+i*50}" y="176" width="44" height="22" rx="3" class="s-ms"/><text x="${200+i*50}" y="191" class="s-ink">#${b}</text>`).join("")}
<text x="20" y="226" class="s-mut">logical token positions map to physical blocks anywhere in the pool; the kernel reads through the table</text>
</svg>`},
hashchain:{cap:"Each full 16-token block's hash covers its parent's hash, so equal hashes mean equal prefixes. Two requests sharing a system prompt reuse the same physical blocks.",svg:`<svg viewBox="0 0 640 200" role="img" aria-label="Chained block hashes">
<rect x="20" y="78" width="70" height="40" rx="4" class="s-p"/><text x="30" y="96" class="s-mut">root</text><text x="30" y="110" class="s-mut">salt</text>
${["h1","h2","h3"].map((h,i)=>`<line x1="${90+i*110}" y1="98" x2="${110+i*110}" y2="98" class="s-axis" stroke-width="1.5"/><rect x="${110+i*110}" y="78" width="90" height="40" rx="4" class="s-ms"/><text x="${122+i*110}" y="96" class="s-ink">${h}</text><text x="${122+i*110}" y="110" class="s-mut">ref 2</text>`).join("")}
<line x1="420" y1="98" x2="450" y2="50" class="s-axis" stroke-width="1.5"/><line x1="420" y1="98" x2="450" y2="146" class="s-axis" stroke-width="1.5"/>
<rect x="450" y="30" width="90" height="40" rx="4" class="s-cs"/><text x="462" y="48" class="s-ink">h4a</text><text x="462" y="62" class="s-mut">ref 1</text>
<rect x="450" y="126" width="90" height="40" rx="4" class="s-as"/><text x="462" y="144" class="s-ink">h4b</text><text x="462" y="158" class="s-mut">ref 1</text>
<text x="552" y="54" class="s-mut">req A</text><text x="552" y="150" class="s-mut">req B</text>
<text x="110" y="150" class="s-mut">h_i = SHA-256(h_(i−1) ‖ 16 token ids)</text><text x="110" y="62" class="s-ink">shared system prompt: computed once</text>
</svg>`},
stall:{cap:"Continuous batching, one decoding user's view. Each A prefill step pauses every running decode: one ITL gap of ~0.26 s. Averaged over 1,000 tokens it barely moves TPOT.",svg:`<svg viewBox="0 0 640 200" role="img" aria-label="Prefill stall in continuous batching">
<text x="20" y="24" class="t-lg s-ink">Engine steps (M4)</text>
${Array.from({length:6},(_,i)=>`<rect x="${20+i*24}" y="36" width="20" height="30" rx="2" class="s-m"/>`).join("")}
<rect x="166" y="36" width="190" height="30" rx="2" class="s-c"/><text x="178" y="56" fill="#fff">A prefill, 8K tokens</text>
${Array.from({length:10},(_,i)=>`<rect x="${362+i*24}" y="36" width="20" height="30" rx="2" class="s-m"/>`).join("")}
<text x="20" y="104" class="t-lg s-ink">B request's tokens</text>
${Array.from({length:6},(_,i)=>`<circle cx="${30+i*24}" cy="122" r="5" class="s-m"/>`).join("")}
${Array.from({length:10},(_,i)=>`<circle cx="${372+i*24}" cy="122" r="5" class="s-m"/>`).join("")}
<line x1="150" y1="140" x2="372" y2="140" class="s-cst" stroke-width="2"/><line x1="150" y1="134" x2="150" y2="146" class="s-cst" stroke-width="2"/><line x1="372" y1="134" x2="372" y2="146" class="s-cst" stroke-width="2"/>
<text x="190" y="160" class="s-c">one ITL gap ≈ 0.26 s</text>
<text x="20" y="188" class="s-mut">ITL p99 sees the gap · TPOT = total decode time ÷ tokens averages it away</text>
</svg>`},
chunked:{cap:"Without chunking, one 8K prefill step stalls every decoding user. With a token budget, the prefill is cut into chunks that share steps with decodes.",svg:`<svg viewBox="0 0 640 210" role="img" aria-label="Chunked prefill timeline">
<text x="20" y="24" class="t-lg s-ink">No chunking (M4)</text>
${Array.from({length:4},(_,i)=>`<rect x="${20+i*26}" y="36" width="22" height="34" rx="2" class="s-m"/>`).join("")}
<rect x="126" y="36" width="230" height="34" rx="2" class="s-c"/><text x="160" y="58" fill="#fff">8K prefill: decodes wait</text>
${Array.from({length:9},(_,i)=>`<rect x="${360+i*26}" y="36" width="22" height="34" rx="2" class="s-m"/>`).join("")}
<text x="20" y="118" class="t-lg s-ink">Chunked, 2,048-token budget (M7)</text>
${Array.from({length:4},(_,i)=>`<rect x="${20+i*26}" y="130" width="22" height="34" rx="2" class="s-m"/>`).join("")}
${Array.from({length:4},(_,i)=>`<rect x="${126+i*66}" y="130" width="40" height="34" rx="2" class="s-c"/><rect x="${168+i*66}" y="130" width="22" height="34" rx="2" class="s-m"/>`).join("")}
${Array.from({length:6},(_,i)=>`<rect x="${392+i*26}" y="130" width="22" height="34" rx="2" class="s-m"/>`).join("")}
<text x="20" y="190" class="s-mut">p99 ITL drops from one whole prefill to one chunk; TTFT for the long prompt rises a little</text>
</svg>`},
spec:{cap:"A cheap drafter proposes k tokens; the target model checks all of them in one pass and keeps the longest correct run plus one token of its own.",svg:`<svg viewBox="0 0 640 190" role="img" aria-label="Speculative decoding">
<text x="20" y="26" class="t-lg s-ink">draft (n-gram or 1B)</text>
${["the","cat","sat","on"].map((t,i)=>`<rect x="${20+i*74}" y="38" width="66" height="30" rx="4" class="s-as"/><text x="${32+i*74}" y="58" class="s-ink">${t}</text>`).join("")}
<text x="340" y="58" class="s-mut">k = 4 guesses, cheap</text>
<text x="20" y="104" class="t-lg s-ink">verify (8B, one forward pass over all 4)</text>
${[["the","m"],["cat","m"],["sat","m"],["a","c"]].map(([t,c],i)=>`<rect x="${20+i*74}" y="116" width="66" height="30" rx="4" class="s-${c}"/><text x="${32+i*74}" y="136" fill="#fff">${t}</text>`).join("")}
<text x="340" y="128" class="s-mut">3 accepted + 1 corrected token</text><text x="340" y="144" class="s-mut">= 4 tokens for one target pass</text>
<text x="20" y="176" class="s-mut">greedy output stays exactly the same as without speculation (G0)</text>
</svg>`},
tp:{cap:"Tensor parallelism splits each layer's matrices across GPUs. Column-split then row-split needs one all-reduce per block: two per transformer layer.",svg:`<svg viewBox="0 0 640 210" role="img" aria-label="Tensor parallel split">
<rect x="20" y="80" width="60" height="50" rx="4" class="s-p"/><text x="32" y="110" class="s-ink">x</text>
${[0,1].map(g=>`<rect x="${130}" y="${40+g*80}" width="110" height="50" rx="4" class="s-ms"/><text x="${140}" y="${62+g*80}" class="s-ink">GPU ${g}: W1 cols</text><text x="${140}" y="${78+g*80}" class="s-mut">half the heads</text>
<rect x="${280}" y="${40+g*80}" width="110" height="50" rx="4" class="s-ms"/><text x="${290}" y="${62+g*80}" class="s-ink">GPU ${g}: W2 rows</text><text x="${290}" y="${78+g*80}" class="s-mut">partial sum</text>
<line x1="80" y1="105" x2="130" y2="${65+g*80}" class="s-axis"/><line x1="240" y1="${65+g*80}" x2="280" y2="${65+g*80}" class="s-axis"/><line x1="390" y1="${65+g*80}" x2="440" y2="105" class="s-axis"/>`).join("")}
<rect x="440" y="80" width="110" height="50" rx="4" class="s-c"/><text x="452" y="102" fill="#fff">all-reduce</text><text x="452" y="118" fill="#fff">NCCL</text>
<line x1="550" y1="105" x2="600" y2="105" class="s-axis"/><text x="580" y="98" class="s-ink">y</text>
<text x="20" y="196" class="s-mut">attention block + MLP block per layer → 2 all-reduces per layer, on the critical path of every step</text>
</svg>`},
pd:{cap:"Prefill/decode disaggregation: a prefill worker builds the KV cache and ships it, block by block, to a decode worker. Decoders never stall on a long prompt.",svg:`<svg viewBox="0 0 640 190" role="img" aria-label="Prefill decode disaggregation">
<rect x="20" y="60" width="90" height="50" rx="4" class="s-p"/><text x="34" y="90" class="s-ink">router</text>
<rect x="160" y="40" width="150" height="90" rx="6" class="s-cs"/><text x="174" y="64" class="t-lg s-c">prefill GPU</text><text x="174" y="84" class="s-ink">compute-bound</text><text x="174" y="100" class="s-mut">big batches of tokens</text>
<rect x="450" y="40" width="160" height="90" rx="6" class="s-ms"/><text x="464" y="64" class="t-lg s-m">decode GPU</text><text x="464" y="84" class="s-ink">memory-bound</text><text x="464" y="100" class="s-mut">many concurrent users</text>
<line x1="110" y1="85" x2="160" y2="85" class="s-axis" stroke-width="1.5"/>
${Array.from({length:5},(_,i)=>`<rect x="${322+i*24}" y="76" width="18" height="18" rx="2" class="s-a"/>`).join("")}
<text x="322" y="68" class="s-ink">KV blocks</text><text x="322" y="112" class="s-mut">NCCL / NIXL</text>
<text x="20" y="168" class="s-mut">pays when transfer time ≪ interference it removes; compare against chunked prefill, not plain M4</text>
</svg>`},
tiers:{cap:"KV tiers on one node. Each step down holds more and moves slower. Fetch beats recompute when bytes ÷ bandwidth < prefill time for those tokens.",svg:`<svg viewBox="0 0 640 220" role="img" aria-label="KV cache tiers">
${[["GPU HBM","~80 GB","~2–3 TB/s","s-ms"],["CPU DRAM","~1–2 TB","~25–50 GB/s (PCIe)","s-as"],["Local NVMe","~8–30 TB","~2–7 GB/s per drive","s-cs"]].map(([n,c,b,cl],i)=>`<rect x="${40+i*30}" y="${20+i*62}" width="${300+i*120}" height="50" rx="5" class="${cl}"/><text x="${56+i*30}" y="${42+i*62}" class="t-lg s-ink">${n}</text><text x="${56+i*30}" y="${60+i*62}" class="s-mut">${c} · ${b}</text>`).join("")}
<text x="470" y="60" class="s-ink">evict idle session</text><text x="470" y="76" class="s-mut">(agent tool call)</text>
<text x="470" y="120" class="s-ink">restore on resume</text>
<text x="40" y="210" class="s-mut">score = recompute cost (∝ prefix length) × reuse probability</text>
</svg>`},
moe:{cap:"Expert parallelism, drawn with Kimi K2's 384 experts over 4 GPUs (96 each). Each token's router picks 8; an all-to-all sends tokens to the GPUs holding them and a second brings results back, every MoE layer.",svg:`<svg viewBox="0 0 640 220" role="img" aria-label="MoE all-to-all">
${[0,1,2,3].map(g=>`<rect x="20" y="${20+g*46}" width="120" height="36" rx="4" class="s-ms"/><text x="30" y="${42+g*46}" class="s-ink">GPU ${g} tokens</text>
<rect x="480" y="${20+g*46}" width="140" height="36" rx="4" class="s-cs"/><text x="490" y="${42+g*46}" class="s-ink">experts ${g*96}–${g*96+95}</text>`).join("")}
${[[0,2],[0,3],[1,0],[1,3],[2,1],[2,2],[3,0],[3,1]].map(([a,b])=>`<line x1="140" y1="${38+a*46}" x2="480" y2="${38+b*46}" class="s-ast" stroke-width="1.2" opacity=".7"/>`).join("")}
<rect x="270" y="80" width="90" height="28" rx="4" class="s-sf"/><text x="278" y="98" class="s-ink">dispatch</text>
<text x="20" y="210" class="s-mut">combine = the same pattern in reverse · stragglers (hot experts) set the step time</text>
</svg>`}
};
