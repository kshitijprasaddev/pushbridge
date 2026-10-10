/* Read-only reconstruction of saved MuJoCo states. This is not a simulator. */
(() => {
  'use strict';
  const data = window.PUSHBRIDGE_DATA;
  if (!data?.precision) return;
  const $ = id => document.getElementById(id);
  const study = data.precision;
  const paired = study.suites.paired.results;
  const labels = {
    source_direct: 'Direct transfer',
    source_contact: 'Contact adapter',
    posttrained: 'Adapter + RL'
  };
  const descriptions = {
    source_direct: 'Use the original two-joint arm policy on the three-joint robot.',
    source_contact: 'Keep the policy weights. Adjust tool clearance for the narrower robot tip.',
    posttrained: 'Use the contact adapter and weights refined with 20,000 target-arm SAC interactions.'
  };
  const phaseLabels = {
    start: ['Initial state', 'Robot and object are reset once.'],
    policy: ['Learned approach', 'SAC chooses actions from object, tool and goal positions.'],
    align: ['Precision feedback: align', 'Position the tool behind the object using known contact geometry.'],
    push: ['Precision feedback: push', 'Smaller pushes as the object nears the target centre.'],
    release: ['Precision feedback: release', 'Retract the tool to remove contact force.'],
    hold: ['Tool stationary', 'Wait for contact to settle.'],
    verify: ['Verify full containment', 'Every hold tick must keep the object centre within 8 mm.']
  };
  let name, trace, result, task = 0, playing = false, lastTick = 0, animation;
  for(let i=0;i<20;i++)$('precision-trial').add(new Option(`Task ${String(i+1).padStart(2,'0')}${i===3?' · includes an RL failure':''}`,i));
  const canvas = $('precision-canvas'), ctx = canvas.getContext('2d');
  const scale = 3500;
  const point = p => [90 + (p[0]-.025)*scale, 665 - (p[1]-.015)*scale];
  const circle = (x, y, radius, fill, stroke, lineWidth = 2) => {
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI*2);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke(); }
  };
  const line = (points, colour, width, dash = []) => {
    if (!points.length) return;
    ctx.beginPath(); ctx.strokeStyle = colour; ctx.lineWidth = width;
    ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.setLineDash(dash);
    points.forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p));
    ctx.stroke(); ctx.setLineDash([]);
  };

  function drawReplay() {
    const step = Number($('precision-time').value);
    const state = trace.state[step], puck = point(state.slice(2)), goal = point(trace.goal);
    const tool = point(state.slice(0,2));
    const distance = Math.hypot(state[2]-trace.goal[0], state[3]-trace.goal[1])*1000;
    const margin = 8-distance, atEnd = step === trace.state.length-1;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle = '#edf3e9'; ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.strokeStyle = '#dce7d5'; ctx.lineWidth = 1;
    for (let x=15; x<canvas.width; x+=35) {
      ctx.beginPath(); ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();
    }
    for (let y=0; y<canvas.height; y+=35) {
      ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke();
    }
    ctx.fillStyle = '#70836b';ctx.font = '13px ui-monospace, monospace';
    ctx.fillText('PLANAR TASK VIEW / ACTUAL-SIZED FOOTPRINTS', 55, 668);
    // Reconstruct the physical arm's planar joints from saved joint angles.
    const joints = [[-.25,.105]];
    let angle = 0, position = [-.25,.105];
    [.24,.17,.13].forEach((length, i) => {
      angle += trace.joint[step][i];
      position = [position[0]+length*Math.cos(angle),position[1]+length*Math.sin(angle)];
      joints.push(position);
    });
    line(joints.map(point), '#9eb4a4', 48);
    line(joints.map(point), '#d7e4d7', 31);
    line(joints.slice(2).map(point), '#2e8267', 30);
    joints.slice(1,-1).forEach(p => circle(...point(p), 24, '#1e4338', '#e6eee2', 7));
    line(trace.state.slice(0,step+1).map(s=>point(s.slice(2))), '#549a72', 3, [6,8]);
    circle(...goal, 70, '#c9e8a6', '#5a8a45', 2);
    ctx.setLineDash([4,5]);circle(...goal,28,null,'#6c964e',1.5);ctx.setLineDash([]);
    line([[goal[0]-8,goal[1]],[goal[0]+8,goal[1]]], '#5b7c46',1.5);
    line([[goal[0],goal[1]-8],[goal[0],goal[1]+8]], '#5b7c46',1.5);
    ctx.font='14px ui-monospace, monospace';ctx.fillStyle='#517044';
    ctx.fillText('GOAL',goal[0]-18,goal[1]-85);
    // The drawn puck radius is exactly 12 mm in the same scale as the target.
    circle(puck[0]+2,puck[1]+4,42,'#2a402b18',null);
    circle(...puck,42,'#d95b4c','#a94239',2);
    circle(...puck,30,'#e57360',null);
    circle(...puck,3,'#fff0da',null);
    circle(...tool,10.5,'#348dc4','#dfeef2',3);
    ctx.font='13px ui-monospace, monospace';ctx.fillStyle='#315c49';
    ctx.fillText('TIP',tool[0]-12,tool[1]+29);
    ctx.fillStyle='#fffef8';ctx.fillRect(779,433,276,235);
    ctx.strokeStyle='#d5e0ce';ctx.lineWidth=1;ctx.strokeRect(779,433,276,235);
    ctx.font='12px ui-monospace, monospace';ctx.fillStyle='#5c7455';ctx.fillText('GOAL CLOSE-UP / 4 PX PER MM',795,458);
    ctx.save();ctx.beginPath();ctx.rect(783,470,267,184);ctx.clip();
    const inset=[917,564];
    circle(...inset,80,'#c9e8a6','#5a8a45',2);
    ctx.setLineDash([4,5]);circle(...inset,32,null,'#6c964e',1);ctx.setLineDash([]);
    circle(inset[0]+(state[2]-trace.goal[0])*4000,inset[1]-(state[3]-trace.goal[1])*4000,48,'#dc6352','#a94239',2);
    ctx.restore();
    $('precision-clock').textContent=`${(step*.1).toFixed(1)} s`;
    $('precision-error').textContent=`${distance.toFixed(1)} mm`;
    $('precision-clearance').textContent=`${margin>=0?'+':''}${margin.toFixed(1)} mm`;
    $('precision-clearance').style.color=margin>=0?'#22653f':'#a15b37';
    const phase=phaseLabels[trace.phase[step]] || phaseLabels.policy;
    $('precision-phase').textContent=phase[0];$('precision-phase-note').textContent=phase[1];
    const badge=$('completion-badge');badge.className='completion-badge';
    if(atEnd&&result.success){badge.textContent='✓ FULLY INSIDE · 1 s HOLD PASSED';badge.classList.add('complete');}
    else if(atEnd){badge.textContent=margin>=0?'INSIDE · HOLD NOT COMPLETED':'OUTSIDE · TASK INCOMPLETE';badge.classList.add('pending');}
    else{badge.textContent=margin>=0?'INSIDE · VERIFICATION PENDING':'PUSHING TOWARD THE TARGET';}
    drawTimeline(step);
  }

  function drawTimeline(step) {
    const plot=$('precision-error-canvas'), c=plot.getContext('2d');
    const base=name.replace('_finish','');
    const alone=data.precision_traces[base][task], finish=data.precision_traces[base+'_finish'][task];
    const errors=t=>t.state.map(s=>Math.hypot(s[2]-t.goal[0],s[3]-t.goal[1])*1000);
    const series=[errors(alone),errors(finish)];
    const length=Math.max(...series.map(a=>a.length))-1;
    const maxError=Math.max(40,...series.flat());
    const xy=(i,error)=>[62+i/length*930,192-error/maxError*160];
    c.clearRect(0,0,plot.width,plot.height);
    c.font='12px ui-monospace, monospace';
    [8,18].forEach(error=>{
      const y=xy(0,error)[1];c.strokeStyle=error===8?'#72945e':'#bea275';c.lineWidth=1;c.setLineDash([5,5]);
      c.beginPath();c.moveTo(62,y);c.lineTo(992,y);c.stroke();c.setLineDash([]);
      c.fillStyle=error===8?'#436a35':'#9b7746';c.fillText(`${error} mm`,1004,y+4);
    });
    series.forEach((values,index)=>{
      c.beginPath();c.strokeStyle=index?'#21805b':'#ba8551';c.lineWidth=2.5;c.lineJoin='round';
      values.forEach((error,i)=>i?c.lineTo(...xy(i,error)):c.moveTo(...xy(i,error)));c.stroke();
    });
    const handoff=finish.phase.findIndex(p=>p!=='policy'&&p!=='start');
    if(handoff>=0){
      const x=xy(handoff,0)[0];c.strokeStyle='#9ab6a0';c.setLineDash([2,4]);
      c.beginPath();c.moveTo(x,28);c.lineTo(x,192);c.stroke();c.setLineDash([]);
      c.fillStyle='#57755b';c.fillText('FEEDBACK STARTS',Math.min(x+8,850),24);
    }
    const cursor=xy(step,0)[0];c.strokeStyle='#213c2e';c.lineWidth=1.5;c.beginPath();c.moveTo(cursor,34);c.lineTo(cursor,192);c.stroke();
    c.fillStyle='#657761';c.fillText('mm',18,31);c.fillText(`${Math.round(maxError)}`,18,48);
    for(let i=0;i<=length;i+=20)c.fillText(`${(i*.1).toFixed(0)} s`,xy(i,0)[0]-8,221);
  }

  function stopReplay() {
    playing=false;cancelAnimationFrame(animation);$('precision-play').textContent='Replay task';
  }
  function playFrame(now) {
    if (!playing) return;
    if (!lastTick) lastTick=now;
    if(now-lastTick>=100){
      const elapsed=Math.floor((now-lastTick)/100);
      $('precision-time').value=Math.min(trace.state.length-1,Number($('precision-time').value)+elapsed);
      lastTick+=elapsed*100;drawReplay();
      if(Number($('precision-time').value)>=trace.state.length-1){stopReplay();return;}
    }
    animation=requestAnimationFrame(playFrame);
  }
  function chooseReplay() {
    stopReplay();
    const strategy=$('precision-strategy').value, seed=$('precision-seed').value;
    const hybrid=document.querySelector('input[name="finish-mode"]:checked').value==='hybrid';
    name=`${strategy}_seed${seed}${hybrid?'_finish':''}`;
    task=Number($('precision-trial').value);
    trace=data.precision_traces[name][task];result=paired[name].runs[task];
    $('precision-task-label').textContent=`Compact 3R · test 2027 · task ${String(task+1).padStart(2,'0')}`;
    $('precision-time').max=trace.state.length-1;$('precision-time').value=trace.state.length-1;
    $('precision-description').textContent=descriptions[strategy];
    $('precision-score').textContent=`${paired[name].successes} / 20`;
    $('precision-score-note').textContent=`${labels[strategy]}, seed ${seed}${hybrid?', with explicit feedback':', policy alone'}. Mean centre error ${paired[name].mean_final_distance_mm.toFixed(1)} mm.`;
    $('precision-trace-link').href=`results/precision_finish/paired/${name}_${String(task).padStart(3,'0')}.npz`;
    const video=$('precision-video');
    video.pause();
    // Native videos cover seed 1; every seed has a saved-state replay above.
    video.src=`results/precision_finish/paired/${strategy}_seed1${hybrid?'_finish':''}.mp4`;
    video.poster=`results/precision_finish/paired/${strategy}_seed1${hybrid?'_finish':''}.jpg`;
    video.load();
    $('native-video-scope').textContent=seed==='1'&&task===0?'SAME STATE TRACE +':'VIDEO: SEED 1, TASK 01 +';
    const note=video.nextElementSibling;
    note.textContent=`Native MuJoCo video: ${labels[strategy]}, seed 1, task 01, ${hybrid?'with precision feedback':'policy alone'}. ${seed==='1'&&task===0?'It renders the same trace as the task view above.':`The task view above is seed ${seed}, task ${String(task+1).padStart(2,'0')}. Native videos cover seed 1, task 01 only; every task has an exact state replay above.`} The camera and drawing do not change object positions.`;
    drawReplay();
  }
  $('precision-strategy').addEventListener('change',chooseReplay);
  $('precision-seed').addEventListener('change',chooseReplay);
  $('precision-trial').addEventListener('change',chooseReplay);
  $('precision-failure').addEventListener('click',()=>{
    $('precision-strategy').value='posttrained';$('precision-seed').value='2';$('precision-trial').value='3';
    document.querySelector('input[name="finish-mode"][value="hybrid"]').checked=true;chooseReplay();
    document.querySelector('.precision-workbench').scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
  });
  document.querySelectorAll('input[name="finish-mode"]').forEach(input=>input.addEventListener('change',chooseReplay));
  $('precision-time').addEventListener('input',()=>{stopReplay();drawReplay();});
  $('precision-play').addEventListener('click',()=>{
    if(playing){stopReplay();return;}
    if(Number($('precision-time').value)>=trace.state.length-1)$('precision-time').value=0;
    playing=true;lastTick=0;$('precision-play').textContent='Pause';drawReplay();animation=requestAnimationFrame(playFrame);
  });
  $('precision-final').addEventListener('click',()=>{stopReplay();$('precision-time').value=trace.state.length-1;drawReplay();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stopReplay();});
  chooseReplay();

  const tbody=$('precision-table').querySelector('tbody');
  Object.entries(labels).forEach(([prefix,label])=>{
    const plain=[0,1,2].map(seed=>paired[`${prefix}_seed${seed}`]);
    const finished=[0,1,2].map(seed=>paired[`${prefix}_seed${seed}_finish`]);
    const row=document.createElement('tr');if(prefix==='source_contact')row.className='highlight';
    [label,plain.map(r=>`${r.successes}/20`).join(' · '),finished.map(r=>`${r.successes}/20`).join(' · '),
      `${(finished.reduce((sum,r)=>sum+r.mean_final_distance_mm,0)/3).toFixed(1)} mm`].forEach(value=>row.append(Object.assign(document.createElement('td'),{textContent:value})));
    tbody.append(row);
  });
  const baseline=paired.feedback_only;
  $('precision-baseline').textContent=`Feedback alone: ${baseline.successes}/20 full fits under the same budget, mean error ${baseline.mean_final_distance_mm.toFixed(1)} mm. Learned approaches usually reach the precision region sooner. The RL seed 2 hybrid still fails one task: a finisher cannot help a policy that never reaches the handoff region.`;
  const fresh=study.suites.fresh.results;
  $('precision-fresh').textContent=`Twenty additional tasks (seed 9092), with seed 1 policies and frozen controller settings: `+
    Object.entries(labels).map(([prefix,label])=>`${label.toLowerCase()} ${fresh[`${prefix}_seed1`].successes}/20 → ${fresh[`${prefix}_seed1_finish`].successes}/20`).join('; ')+
    `. Feedback alone: ${fresh.feedback_only.successes}/20. An extra check on this simulation task, with no additional training.`;

  function drawAudit() {
    const tolerance=Number($('audit-tolerance').value), canvas=$('audit-canvas'), c=canvas.getContext('2d');
    c.clearRect(0,0,canvas.width,canvas.height);
    const goal=[285,200], scale=6, puck=[goal[0]+tolerance*scale,goal[1]];
    c.fillStyle='#f2f5eb';c.fillRect(0,0,canvas.width,canvas.height);
    c.beginPath();c.arc(...goal,20*scale,0,Math.PI*2);c.fillStyle='#c9e8a6';c.fill();c.strokeStyle='#5a8a45';c.lineWidth=2;c.stroke();
    c.beginPath();c.arc(...puck,12*scale,0,Math.PI*2);c.fillStyle='#dc6352';c.fill();c.strokeStyle='#a94239';c.stroke();
    c.setLineDash([4,5]);c.beginPath();c.arc(...goal,8*scale,0,Math.PI*2);c.strokeStyle='#66884c';c.stroke();c.setLineDash([]);
    c.font='16px ui-monospace, monospace';c.fillStyle='#345c40';
    c.fillText('TARGET RADIUS 20 mm',170,46);c.fillText('PUCK RADIUS 12 mm',417,80);
    c.beginPath();c.moveTo(goal[0],335);c.lineTo(puck[0],335);c.strokeStyle='#576e4e';c.stroke();
    c.fillText(`CENTRE DISTANCE ${tolerance} mm`,200,365);
    c.font='bold 16px ui-monospace, monospace';c.fillStyle=tolerance<=8?'#216a45':'#9c5633';
    c.fillText(tolerance<=8?'FULLY INSIDE':`EXTENDS ${tolerance-8} mm OUTSIDE`,419,290);
    $('audit-label').textContent=`${tolerance} mm`;
    $('audit-counts').replaceChildren();
    Object.entries(labels).forEach(([prefix,label])=>{
      const values=study.original_endpoint_distances_mm[`${prefix}_seed1`];
      const item=document.createElement('div');
      item.append(Object.assign(document.createElement('strong'),{textContent:`${values.filter(n=>n<=tolerance).length}/20`}),Object.assign(document.createElement('span'),{textContent:label}));
      $('audit-counts').append(item);
    });
  }
  $('audit-tolerance').addEventListener('input',drawAudit);
  document.querySelectorAll('[data-tolerance]').forEach(button=>button.addEventListener('click',()=>{$('audit-tolerance').value=button.dataset.tolerance;drawAudit();}));
  drawAudit();
})();
