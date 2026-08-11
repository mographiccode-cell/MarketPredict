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

  async function initPredictionForm(){
    const form=$('#prediction-form'); if(!form) return;
    let schema;
    try{schema=await fetch('/api/schema').then(r=>r.json());}catch(e){return;}
    const objective=$('#Campaign_Objective'), platform=$('#Platform'), placement=$('#Placement'), content=$('#Content_Type'), region=$('#Region'), city=$('#City'), bidding=$('#Bidding_Strategy');
    function updatePlatform(){const p=platform?.value;const oldP=placement?.value,oldC=content?.value;setOptions(placement,schema.platform_placements[p]||[],oldP);setOptions(content,schema.platform_content_types[p]||[],oldC);}
    function updateRegion(){const r=region?.value,old=city?.value;setOptions(city,schema.region_city[r]||[],old);}
    function updateObjective(){
      const o=objective?.value, old=bidding?.value; setOptions(bidding,schema.objective_bidding_strategies[o]||[],old);
      $$('.objective-block').forEach(block=>{const show=block.dataset.objective===o;block.hidden=!show;$$('.objective-input',block).forEach(input=>input.required=show);});
      const active=$$('.form-progress span');active.forEach((s,i)=>s.classList.toggle('active',i===0));
    }
    platform?.addEventListener('change',updatePlatform);region?.addEventListener('change',updateRegion);objective?.addEventListener('change',updateObjective);
    updatePlatform();updateRegion();updateObjective();
    const start=$('#Start_Date'); if(start&&!start.value){const d=new Date();start.value=d.toISOString().slice(0,10);}
  }
  initPredictionForm();

  $$('.rubric-select').forEach(sel=>{
    const update=()=>{const opt=sel.options[sel.selectedIndex];const desc=sel.closest('.rubric-card')?.querySelector('.rubric-desc');if(desc)desc.textContent=opt?.dataset?.desc||'اختر مستوى لعرض تعريفه.';};
    sel.addEventListener('change',update);update();
  });

  const printBtn=$('#print-result'); if(printBtn) printBtn.addEventListener('click',()=>window.print());
})();
