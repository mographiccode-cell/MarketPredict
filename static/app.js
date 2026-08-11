(function(){
  const $=(sel,root=document)=>root.querySelector(sel);
  const $$=(sel,root=document)=>Array.from(root.querySelectorAll(sel));
  const menu=$('#menu-toggle'),nav=$('#main-nav');
  if(menu&&nav) menu.addEventListener('click',()=>nav.classList.toggle('open'));

  function setOptions(select, options, previous){
    if(!select) return;
    const placeholder=select.querySelector('option[value=""]')?.textContent || 'اختر';
    select.innerHTML='';
    const ph=document.createElement('option'); ph.value=''; ph.textContent=placeholder; select.appendChild(ph);
    (options||[]).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;if(v===previous)o.selected=true;select.appendChild(o);});
  }

  function updateEvidenceCard(card){
    if(!card) return;
    const rows=$$('.evidence-row',card);
    const checked=rows.map(row=>$('input[type="radio"]:checked',row)).filter(Boolean);
    const yes=checked.filter(input=>input.value==='1').length;
    const output=$('.evidence-score',card);
    const complete=rows.length>0 && checked.length===rows.length;
    card.classList.toggle('complete',complete);
    if(!output) return;
    if(!complete){
      output.textContent=`أجبت عن ${checked.length} من ${rows.length}. الدرجة لن تُعتمد حتى تكتمل الأدلة.`;
      return;
    }
    const score=Math.round((30+60*(yes/rows.length))*10)/10;
    output.textContent=`درجة الجاهزية المشتقة: ${score}/100 — محسوبة من ${yes} أدلة مثبتة من ${rows.length}.`;
  }

  $$('.evidence-card').forEach(card=>{
    $$('input[type="radio"]',card).forEach(input=>input.addEventListener('change',()=>updateEvidenceCard(card)));
    updateEvidenceCard(card);
  });

  async function initPredictionForm(){
    const form=$('#prediction-form'); if(!form) return;
    let schema;
    try{
      const response=await fetch('/api/schema',{headers:{'Accept':'application/json'}});
      if(!response.ok) throw new Error('schema request failed');
      schema=await response.json();
    }catch(e){
      const dock=$('.submit-dock small',form); if(dock) dock.textContent='تعذر تحميل قيود النموذج. أعد تحميل الصفحة قبل إرسال الحملة.';
      return;
    }
    const objective=$('#Campaign_Objective'), platform=$('#Platform'), placement=$('#Placement'), content=$('#Content_Type'), region=$('#Region'), city=$('#City'), bidding=$('#Bidding_Strategy');
    function updatePlatform(){const p=platform?.value;const oldP=placement?.value,oldC=content?.value;setOptions(placement,schema.platform_placements[p]||[],oldP);setOptions(content,schema.platform_content_types[p]||[],oldC);}
    function updateRegion(){const r=region?.value,old=city?.value;setOptions(city,schema.region_city[r]||[],old);}
    function updateObjective(){
      const o=objective?.value, old=bidding?.value; setOptions(bidding,schema.objective_bidding_strategies[o]||[],old);
      $$('.objective-block').forEach(block=>{
        const show=block.dataset.objective===o;
        block.hidden=!show;
        $$('.objective-input',block).forEach(input=>{
          input.required=show;
          input.disabled=!show;
        });
      });
    }
    platform?.addEventListener('change',updatePlatform);region?.addEventListener('change',updateRegion);objective?.addEventListener('change',updateObjective);
    updatePlatform();updateRegion();updateObjective();
    const start=$('#Start_Date'); if(start&&!start.value){const d=new Date();start.value=d.toISOString().slice(0,10);}
  }
  initPredictionForm();

  const printBtn=$('#print-result'); if(printBtn) printBtn.addEventListener('click',()=>window.print());
})();
