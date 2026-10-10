(() => {
  'use strict';
  const D = window.PUSHBRIDGE_DATA;
  const $ = id => document.getElementById(id);
  if (!D) { $('main').prepend(Object.assign(document.createElement('p'), {textContent: 'Dashboard data is missing. Run python scripts/build_dashboard.py from the project directory.'})); return; }
  const mm = n => `${n.toFixed(1)} mm`;
  const fmt = n => `${n.toFixed(1)}×`;
  $('hero-speed').textContent = fmt(D.speed.torch_direct_p50_speedup);
  $('exact-speed').textContent = fmt(D.speed.torch_direct_p50_speedup);
  $('numpy-speed').textContent = `${fmt(D.speed.p50_speedup)} faster with NumPy`;
  $('numpy-note').textContent = `All 20 success outcomes match. Floating-point changes produce up to ${D.speed.max_trace_position_vector_difference_mm.toFixed(1)} mm of tool/object state-vector difference; this path is not bit-identical.`;
  if (!D.speed.torch_direct_bit_identical_all_actions_and_states) {
    document.querySelector('.verification p').textContent = 'Trace agreement failed. Inspect the recorded metrics before using this actor.';
  }
  const roleText = {train: 'TRAINING CLIP', validation: 'VALIDATION CLIP', test: 'HELD-OUT TEST CLIP'};
  D.clips.forEach((c, i) => $('clip-select').add(new Option(`${c.source} · ${c.role}`, i)));
  $('clip-select').value = '1';

  function plot(canvas, dark = false) {
    const ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const box = {x: 68, y: 35, width: w - 108, height: h - 105};
    const xy = p => [box.x + box.width * p[0]/.297, box.y + box.height * (1-p[1]/.21)];
    ctx.font = '14px ui-monospace, monospace';
    ctx.lineWidth = 1;
    for (let x = 0; x < .3; x += .05) {
      const px = xy([x, 0])[0];
      ctx.strokeStyle = dark ? '#3a5147' : '#e2e7dd';
      ctx.beginPath(); ctx.moveTo(px, box.y); ctx.lineTo(px, box.y+box.height); ctx.stroke();
      ctx.fillStyle = dark ? '#a8c0af' : '#7c8b7d'; ctx.fillText(`${Math.round(x*1000)}`, px-10, box.y+box.height+25);
    }
    for (let y = 0; y < .22; y += .05) {
      const py = xy([0, y])[1];
      ctx.strokeStyle = dark ? '#3a5147' : '#e2e7dd';
      ctx.beginPath(); ctx.moveTo(box.x, py); ctx.lineTo(box.x+box.width, py); ctx.stroke();
      ctx.fillStyle = dark ? '#a8c0af' : '#7c8b7d'; ctx.fillText(`${Math.round(y*1000)}`, 21, py+5);
    }
    const goal = xy([.225, .105]);
    ctx.strokeStyle = dark ? '#789b7b' : '#a5bc94';
    ctx.beginPath(); ctx.ellipse(goal[0], goal[1], box.width*.02/.297, box.height*.02/.21, 0, 0, 2*Math.PI); ctx.stroke();
    ctx.fillStyle = dark ? '#a8c0af' : '#7c8b7d';
    ctx.fillText('normalised task frame / mm', box.x+box.width/2-115, h-15);
    function path(points, color, width = 3, point = true) {
      if (!points.length) return;
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round';
      ctx.beginPath(); points.forEach((p, i) => { const q = xy(p); i ? ctx.lineTo(...q) : ctx.moveTo(...q); }); ctx.stroke();
      if (point) { const q=xy(points[points.length-1]); ctx.fillStyle=color; ctx.beginPath();ctx.arc(...q,6,0,2*Math.PI);ctx.fill(); }
    }
    return {ctx, xy, path};
  }

  function setClip() {
    const c = D.clips[Number($('clip-select').value)], video = $('clip-video');
    video.onloadedmetadata=()=>{video.currentTime=c.retained_s[0];};
    video.pause(); video.poster=c.poster; video.src=c.video; video.load();
    $('clip-role').textContent=roleText[c.role];
    $('clip-time').textContent=`Retained ${c.retained_s[0].toFixed(1)}-${c.retained_s[1].toFixed(1)} s`;
    $('clip-transitions').textContent=c.transitions;
    $('clip-moving').textContent=c.moving_transitions;
    const fig=plot($('clip-canvas'));
    fig.path(c.state.map(s=>s.slice(0,2)), '#bd854d', 2, false);
    fig.path(c.state.map(s=>s.slice(2)), '#219a83', 4);
    const q=fig.xy(c.state[0].slice(2));fig.ctx.strokeStyle='#219a83';fig.ctx.lineWidth=2;fig.ctx.beginPath();fig.ctx.arc(...q,7,0,2*Math.PI);fig.ctx.stroke();
  }
  $('clip-select').addEventListener('change',setClip); setClip();
  const heroVideo=$('hero-human');
  const seekHero=()=>{heroVideo.currentTime=D.clips[1].retained_s[0];};
  if(heroVideo.readyState>0)seekHero();else heroVideo.addEventListener('loadedmetadata',seekHero,{once:true});

  const T=D.transfer.results;
  function setTransfer() {
    const strategy=document.querySelector('input[name="strategy"]:checked').value, seed=$('seed-select').value;
    const name=`${strategy}_seed${seed}`, r=T[name], video=$('transfer-video');
    video.pause();video.poster=`results/embodiment_transfer/${name}.jpg`;video.src=`results/embodiment_transfer/${name}.mp4`;video.load();
    $('transfer-success').textContent=`${r.successes} / ${r.episodes}`;
    $('transfer-error').textContent=mm(r.mean_final_distance_mm);
    const first=r.runs[0];
    $('transfer-caption').textContent=`Original first trial: ${first.success?'passes the 18 mm centre rule':'fails the 18 mm centre rule'}, ${mm(first.final_distance_mm)} final distance. ${first.final_distance_mm<=8?'The whole object fits at this endpoint.':'The whole object does not fit inside the circle.'} The stricter experiment above preserves these original results.`;
  }
  document.querySelectorAll('input[name="strategy"]').forEach(i=>i.addEventListener('change',setTransfer));
  $('seed-select').addEventListener('change',setTransfer);setTransfer();
  const tbody=$('transfer-table').querySelector('tbody');
  const rows=[['Direct source SAC','source_direct'],['Frozen SAC + contact adapter','source_contact'],['Adapter + target RL','posttrained']];
  rows.forEach(([label,prefix])=>{
    const values=[0,1,2].map(s=>T[`${prefix}_seed${s}`]);
    const tr=document.createElement('tr');if(prefix==='source_contact')tr.className='highlight';
    [label,...values.map(r=>`${r.successes}/${r.episodes}`),mm(values.reduce((sum,r)=>sum+r.mean_final_distance_mm,0)/3)].forEach(v=>tr.append(Object.assign(document.createElement('td'),{textContent:v})));tbody.append(tr);
  });
  [['Recorded action replay','replay'],['Source classical feedback','geometry'],['Source feedback + adapter','geometry_contact'],['Compact-aware classical feedback*','geometry_compact'],['Learned-model MPC + adapter','mpc_contact']].forEach(([label,name])=>{
    const r=T[name];if(!r)return;const tr=document.createElement('tr');[label,`${r.successes}/${r.episodes}`,'N/A','N/A',mm(r.mean_final_distance_mm)].forEach(v=>tr.append(Object.assign(document.createElement('td'),{textContent:v})));tbody.append(tr);
  });
  const tableNote=document.createElement('p');tableNote.className='small-note';tableNote.textContent='Baselines each have one 20-task run. *Compact-aware feedback is a follow-up baseline developed after the initial comparison and checked on validation tasks. All earlier results are retained.';$('transfer-table').parentElement.after(tableNote);

  [['Imitation only','bc_only'],['SAC from scratch','sac_scratch'],['Personal demos → SAC','sac_demo_initialized']].forEach(([label,prefix])=>{
    const row=document.createElement('div');row.className='rl-row'+(prefix==='sac_demo_initialized'?' good':'');row.append(Object.assign(document.createElement('span'),{textContent:label}),Object.assign(document.createElement('strong'),{textContent:[0,1,2].map(s=>`${D.rl[`${prefix}_seed${s}`].successes}/20`).join(' · ')}));$('rl-results').append(row);
  });
  const postCounts=[0,1,2].map(seed=>{
    const before=D.rl[`sac_scratch_seed${seed}`],after=D.personal_posttraining.results[`personal_posttrained_seed${seed}`];
    const item=document.createElement('div');item.append(Object.assign(document.createElement('span'),{textContent:`SEED ${seed}`}),Object.assign(document.createElement('strong'),{textContent:`${before.successes}/20 → ${after.successes}/20`}),Object.assign(document.createElement('small'),{textContent:`Mean final error ${mm(before.mean_final_distance_mm)} → ${mm(after.mean_final_distance_mm)}`}));$('personal-post-results').append(item);return after.successes-before.successes;
  });
  $('personal-post-finding').textContent=`At this fixed supervised budget, success counts improve for ${postCounts.filter(n=>n>0).length} of the three seeds and decline for ${postCounts.filter(n=>n<0).length}. This tests the timing of personal-data use; the human labels are not equivalent to robot transitions.`;
  D.world.windows.forEach((w,i)=>$('window-select').add(new Option(`Window ${i+1} · ${w.video_time_s[0].toFixed(1)} s`,i)));
  function setWorld(syncVideo=true) {
    const i=Number($('window-select').value), step=Number($('world-step').value), win=D.world.windows[i];
    const fig=plot($('world-canvas'),true);
    fig.path(win.tool.slice(0,step+1),'#7e9189',1,false);
    fig.path(win.analytic.slice(0,step+1),'#c28b51',3);
    fig.path(win.predicted.slice(0,step+1),'#55afdc',4);
    fig.path(win.observed.slice(0,step+1),'#55c398',3);
    const dist=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1])*1000;
    $('world-error').textContent=mm(dist(win.predicted[step],win.observed[step]));
    $('world-zero').textContent=mm(dist(win.observed[0],win.observed[step]));
    $('step-label').textContent=`${step} / 10`;
    if(syncVideo){const video=$('world-video');video.pause();if(video.readyState>0)video.currentTime=(i*11+step)/10;}
  }
  $('window-select').addEventListener('change',()=>setWorld());$('world-step').addEventListener('input',()=>setWorld());setWorld(false);
  $('world-video').addEventListener('loadedmetadata',()=>setWorld());
  $('world-video').addEventListener('timeupdate',()=>{
    if($('world-video').paused)return;const frame=Math.floor($('world-video').currentTime*10),i=Math.min(D.world.windows.length-1,Math.floor(frame/11));$('window-select').value=i;$('world-step').value=Math.min(10,frame%11);setWorld(false);
  });
  const domains=[['Held-out human recording',D.prediction.test,'validation_scaled_radius_coverage'],['Source robot',D.world.robot_domain_checks.source_robot,'one_step_empirical_radius_coverage'],['Compact robot',D.world.robot_domain_checks.compact_robot,'one_step_empirical_radius_coverage']];
  domains.forEach(([label,r,coverage])=>{if(!r)return;const row=document.createElement('div');row.className='domain-row';const text=document.createElement('span');text.textContent=label;text.append(Object.assign(document.createElement('small'),{textContent:`${r.transitions} transitions · ${(100*r[coverage]).toFixed(1)}% empirical radius coverage`}));row.append(text,Object.assign(document.createElement('b'),{textContent:mm(r.mean_error_mm)}));$('domain-checks').append(row);});
  [['sb3_predict','Standard SB3 prediction','Full deterministic predict wrapper'],['torch_direct','Direct Torch actor','Same kernels, no training/distribution wrapper'],['numpy_actor','NumPy actor','Same float32 weights, different arithmetic order']].forEach(([key,label,note])=>{
    const r=D.speed.latency[key],row=document.createElement('div');row.className='speed-row';const top=document.createElement('div');top.className='speed-bar-label';top.append(Object.assign(document.createElement('span'),{textContent:label}),Object.assign(document.createElement('b'),{textContent:`${r.p50_ms.toFixed(3)} ms`}));const track=document.createElement('div');track.className='bar-track';const fill=document.createElement('div');fill.className='bar-fill';fill.style.width=`${100*r.p50_ms/D.speed.latency.sb3_predict.p50_ms}%`;track.append(fill);row.append(top,track,Object.assign(document.createElement('small'),{textContent:`p95 ${r.p95_ms.toFixed(3)} ms · ${note}`}));$('speed-bars').append(row);
  });
  const failure=D.transfer.verified_failures.find(f=>f.mode==='posttrained_seed2');
  if(failure){$('post-failure-video').src=`results/embodiment_transfer/${failure.video}`;$('post-failure-video').poster=`results/embodiment_transfer/${failure.poster}`;}
  $('copy-command').addEventListener('click',async()=>{
    try { await navigator.clipboard.writeText($('run-commands').innerText);$('copy-command').textContent='Copied'; }
    catch { const range=document.createRange();range.selectNodeContents($('run-commands'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);$('copy-command').textContent='Commands selected'; }
    setTimeout(()=>$('copy-command').textContent='Copy commands',2500);
  });
  const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(!entry.isIntersecting)return;document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.hash===`#${entry.target.id}`));}),{rootMargin:'-15% 0px -65% 0px'});
  document.querySelectorAll('nav a').forEach(a=>observer.observe(document.querySelector(a.hash)));
})();
