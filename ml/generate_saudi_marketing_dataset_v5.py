import numpy as np, pandas as pd, json, hashlib
from pathlib import Path

SEED=20260811; N=72000; N_BRANDS=1100
rng=np.random.default_rng(SEED)
START=pd.Timestamp('2022-01-01'); END=pd.Timestamp('2026-08-10')
GASTAT_POP=35_300_280; INTERNET=.996
regions=np.array(['Riyadh','Makkah','Eastern Region','Madinah','Aseer','Jazan','Qassim','Tabuk','Hail','Najran','Al Jouf','Northern Borders','Al Bahah'])
rw=np.array([.2637,.2501,.1600,.0672,.0613,.0441,.0419,.0281,.0236,.0189,.0184,.0118,.0110]); rw/=rw.sum()
city_map={'Riyadh':'Riyadh','Makkah':'Jeddah','Eastern Region':'Dammam','Madinah':'Madinah','Aseer':'Abha','Jazan':'Jazan','Qassim':'Buraidah','Tabuk':'Tabuk','Hail':'Hail','Najran':'Najran','Al Jouf':'Sakaka','Northern Borders':'Arar','Al Bahah':'Al Bahah'}
industries=np.array(['Retail & E-commerce','Food & Restaurants','Beauty & Fashion','Real Estate','Education','Healthcare','Travel & Hospitality','Automotive','Finance','Telecom','Technology/SaaS','Entertainment'])
iw=np.array([.16,.12,.11,.09,.08,.08,.08,.07,.06,.05,.06,.04])
objectives=np.array(['Sales','Traffic','Awareness','Lead Generation','Engagement','App Installs'])
obj_probs={'Retail & E-commerce':[.38,.14,.12,.08,.16,.12],'Food & Restaurants':[.32,.18,.14,.06,.22,.08],'Beauty & Fashion':[.30,.14,.14,.08,.22,.12],'Real Estate':[.10,.11,.16,.43,.05,.15],'Education':[.09,.13,.17,.34,.09,.18],'Healthcare':[.09,.12,.18,.34,.07,.20],'Travel & Hospitality':[.25,.16,.18,.13,.13,.15],'Automotive':[.13,.12,.18,.35,.05,.17],'Finance':[.07,.10,.16,.29,.05,.33],'Telecom':[.15,.12,.17,.14,.10,.32],'Technology/SaaS':[.16,.11,.15,.28,.07,.23],'Entertainment':[.18,.16,.20,.06,.24,.16]}
platforms=np.array(['Google Search','Instagram','TikTok','Snapchat','YouTube','X','Facebook','LinkedIn'])
base_cpm={'Google Search':52,'Instagram':28,'TikTok':22,'Snapchat':20,'YouTube':28,'X':25,'Facebook':27,'LinkedIn':50}
base_ctr={'Google Search':.035,'Instagram':.011,'TikTok':.014,'Snapchat':.012,'YouTube':.007,'X':.009,'Facebook':.010,'LinkedIn':.007}
base_cvr={'Google Search':.025,'Instagram':.018,'TikTok':.017,'Snapchat':.016,'YouTube':.012,'X':.014,'Facebook':.018,'LinkedIn':.023}
base_er={'Google Search':.004,'Instagram':.024,'TikTok':.032,'Snapchat':.020,'YouTube':.016,'X':.018,'Facebook':.020,'LinkedIn':.012}
fit_table={'Sales':{'Google Search':1.16,'Instagram':1.05,'TikTok':1.00,'Snapchat':.98,'YouTube':.90,'X':.88,'Facebook':.94,'LinkedIn':.92},'Traffic':{'Google Search':1.15,'Instagram':1.03,'TikTok':1.01,'Snapchat':.98,'YouTube':.95,'X':1.00,'Facebook':.95,'LinkedIn':.94},'Awareness':{'Google Search':.88,'Instagram':1.06,'TikTok':1.10,'Snapchat':1.09,'YouTube':1.14,'X':1.01,'Facebook':.96,'LinkedIn':.92},'Lead Generation':{'Google Search':1.15,'Instagram':.98,'TikTok':.91,'Snapchat':.90,'YouTube':.90,'X':.95,'Facebook':.97,'LinkedIn':1.12},'Engagement':{'Google Search':.80,'Instagram':1.12,'TikTok':1.15,'Snapchat':1.08,'YouTube':1.00,'X':1.04,'Facebook':.98,'LinkedIn':.90},'App Installs':{'Google Search':1.03,'Instagram':1.06,'TikTok':1.12,'Snapchat':1.10,'YouTube':1.01,'X':.92,'Facebook':.97,'LinkedIn':.82}}

# brand table
brand_ids=np.array([f'KSA-V5-BRAND-{i+1:04d}' for i in range(N_BRANDS)])
b_ind=rng.choice(industries,N_BRANDS,p=iw); b_home=rng.choice(regions,N_BRANDS,p=rw)
b_quality=np.clip(rng.normal(64,12,N_BRANDS),25,94); b_creative=np.clip(b_quality+rng.normal(0,7,N_BRANDS),20,98); b_landing=np.clip(b_quality+rng.normal(0,7,N_BRANDS),20,98); b_tracking=np.clip(b_quality+rng.normal(0,8,N_BRANDS),15,99); b_local=np.clip(b_quality+rng.normal(4,7,N_BRANDS),20,99); b_trust=np.clip(b_quality+rng.normal(2,7,N_BRANDS),20,99); b_budget=np.exp(rng.normal(np.log(18000),.55,N_BRANDS))*np.clip(b_quality/60,.6,1.7)
brands=pd.DataFrame({'Brand_ID':brand_ids,'Industry':b_ind,'Home_Region':b_home,'Brand_Base_Quality':np.round(b_quality,1),'Typical_Budget_SAR':np.round(b_budget,2)})
prop=np.clip(b_budget/np.median(b_budget),.35,3.0)*rng.lognormal(0,.22,N_BRANDS); prop/=prop.sum()
brand_idx=rng.choice(np.arange(N_BRANDS),N,p=prop); brand_id_rows=brand_ids[brand_idx]; industry=b_ind[brand_idx]

# dates
days=(END-START).days; pool=np.arange(days+1); tw=np.linspace(.82,1.18,days+1); tw/=tw.sum(); date=START+pd.to_timedelta(rng.choice(pool,N,p=tw),unit='D')
objective=np.array([rng.choice(objectives,p=obj_probs[i]) for i in industry])
age=rng.choice(['18-24','25-34','35-44','45-54','55+'],N,p=[.19,.34,.25,.14,.08]); gender=rng.choice(['All','Male','Female'],N,p=[.62,.19,.19]); region=rng.choice(regions,N,p=rw); city=np.array([city_map[x] for x in region])

# platform helper simplified but objective/age/industry aware
fit_weights={'Sales':[2.1,1.55,1.25,1.2,.7,.6,.8,.45],'Traffic':[1.9,1.3,1.05,1,.8,1,.8,.5],'Awareness':[.55,1.3,1.5,1.5,1.9,1,.65,.45],'Lead Generation':[1.9,1,.65,.65,.55,.7,.8,1.5],'Engagement':[.35,1.75,1.8,1.4,1,1,.7,.35],'App Installs':[1,1.3,1.6,1.5,1,.55,.65,.2]}
platform=[]
for o,a,i in zip(objective,age,industry):
    w=np.array(fit_weights[o],float)*np.array([1,1,1,1,1,.65,.55,.35])
    if a=='18-24': w*=np.array([.75,1.15,1.45,1.45,1.05,.9,.4,.25])
    elif a=='25-34': w*=np.array([1,1.25,1.25,1.15,1,.95,.6,.8])
    elif a in ['45-54','55+']: w*=np.array([1.3,.8,.5,.65,1.1,1,1.35,1.2])
    if i in ['Finance','Technology/SaaS','Real Estate','Education']: w*=np.array([1.2,1,1,1,1,1,1,1.5])
    platform.append(rng.choice(platforms,p=w/w.sum()))
platform=np.array(platform)

astrat=[]; intent=[]
sp={'Sales':[.10,.22,.25,.31,.12],'Traffic':[.26,.34,.19,.13,.08],'Awareness':[.48,.31,.11,.04,.06],'Lead Generation':[.09,.26,.28,.22,.15],'Engagement':[.34,.37,.17,.06,.06],'App Installs':[.24,.29,.22,.16,.09]}
for o in objective: astrat.append(rng.choice(['Broad','Interest','Lookalike','Retargeting','Custom'],p=sp[o]))
astrat=np.array(astrat)
ip={'Broad':[.69,.25,.06],'Interest':[.43,.45,.12],'Lookalike':[.30,.52,.18],'Retargeting':[.08,.35,.57],'Custom':[.15,.48,.37]}
for a in astrat: intent.append(rng.choice(['Cold','Warm','Hot'],p=ip[a]))
intent=np.array(intent)

# seasons vectorized via function
season=[]
for d in date:
    y=d.year; s='Normal'; windows={2023:[('Ramadan','2023-03-23','2023-04-20'),('Eid Al-Fitr','2023-04-21','2023-04-24'),('Eid Al-Adha','2023-06-27','2023-07-01')],2024:[('Ramadan','2024-03-11','2024-04-09'),('Eid Al-Fitr','2024-04-10','2024-04-13'),('Eid Al-Adha','2024-06-15','2024-06-19')],2025:[('Ramadan','2025-03-01','2025-03-29'),('Eid Al-Fitr','2025-03-30','2025-04-02'),('Eid Al-Adha','2025-06-05','2025-06-09')],2026:[('Ramadan','2026-02-18','2026-03-19'),('Eid Al-Fitr','2026-03-20','2026-03-23'),('Eid Al-Adha','2026-05-26','2026-05-30')]}
    for name,a,b in windows.get(y,[]):
        if pd.Timestamp(a)<=d<=pd.Timestamp(b): s=name; break
    if s=='Normal' and d.month==2 and 20<=d.day<=24:s='Founding Day'
    if s=='Normal' and d.month==9 and 20<=d.day<=25:s='National Day'
    if s=='Normal' and ((d.month==8 and d.day>=15) or (d.month==9 and d.day<=7)):s='Back to School'
    if s=='Normal' and d.month==11 and d.day>=20:s='White Friday'
    season.append(s)
season=np.array(season)
season_mult=pd.Series(season).map({'Normal':1.,'Ramadan':1.10,'Eid Al-Fitr':1.10,'Eid Al-Adha':1.06,'National Day':1.08,'Founding Day':1.04,'Back to School':1.07,'White Friday':1.13}).to_numpy()

# general setup
duration=np.clip(np.array([rng.normal({'Sales':21,'Traffic':18,'Awareness':28,'Lead Generation':25,'Engagement':18,'App Installs':21}[o],7) for o in objective]),5,60).astype(int)
ind_mult=pd.Series(industry).map({'Retail & E-commerce':1.,'Food & Restaurants':.68,'Beauty & Fashion':.82,'Real Estate':1.55,'Education':.9,'Healthcare':1.05,'Travel & Hospitality':1.18,'Automotive':1.35,'Finance':1.45,'Telecom':1.4,'Technology/SaaS':1.1,'Entertainment':.9}).to_numpy(); obj_mult=pd.Series(objective).map({'Sales':1.2,'Traffic':.8,'Awareness':1.25,'Lead Generation':1.1,'Engagement':.75,'App Installs':1.0}).to_numpy(); budget=np.clip(b_budget[brand_idx]*ind_mult*obj_mult*rng.lognormal(0,.42,N),1500,350000)
# maturity proxy from brand quality/history potential, without using ID
maturity=np.where(b_quality[brand_idx]>=72,'Established',np.where(b_quality[brand_idx]>=56,'Emerging','New'))

# latent true campaign factors
season_bonus=np.where(np.isin(season,['Ramadan','White Friday','National Day','Back to School']),5,0)
true_creative=np.clip(b_creative[brand_idx]+rng.normal(0,9,N)+season_bonus*.25,5,99); true_landing=np.clip(b_landing[brand_idx]+rng.normal(0,8,N),5,99); true_tracking=np.clip(b_tracking[brand_idx]+rng.normal(0,6,N),5,99); true_local=np.clip(b_local[brand_idx]+rng.normal(0,7,N),5,99); true_trust=np.clip(b_trust[brand_idx]+rng.normal(0,7,N),5,99); true_brand=b_quality[brand_idx]
discount=np.zeros(N); commercial=np.isin(objective,['Sales','Traffic','Lead Generation','App Installs']); discount[commercial]=np.clip(rng.gamma(2,5,commercial.sum()),0,40); true_offer=np.clip(48+.8*discount+.2*(true_brand-55)+rng.normal(0,10,N)+np.where(intent=='Hot',6,0),5,99); true_mobile=np.clip(64+.22*(true_brand-55)+rng.normal(0,10,N),5,99); true_schedule=np.clip(64+season_bonus+rng.normal(0,11,N),5,99)
fit=np.array([fit_table[o][p] for o,p in zip(objective,platform)]); intent_score=pd.Series(intent).map({'Cold':42,'Warm':66,'Hot':86}).to_numpy(); strat_score=pd.Series(astrat).map({'Broad':48,'Interest':61,'Lookalike':72,'Retargeting':84,'Custom':78}).to_numpy(); true_fit=np.clip(.28*true_creative+.18*true_local+.20*intent_score+.12*strat_score+24*np.clip((fit-.8)/.4,0,1)+rng.normal(0,6,N),5,99); true_trend=np.clip(54+np.where(np.isin(platform,['TikTok','Instagram','Snapchat']) & np.isin(age,['18-24','25-34']),8,0)+np.where(season!='Normal',6,0)+rng.normal(0,10,N),5,99); true_comp=np.clip(48+np.where(np.isin(industry,['Real Estate','Retail & E-commerce','Beauty & Fashion','Finance','Automotive']),15,0)+np.where(np.isin(platform,['Google Search','Instagram','TikTok','Snapchat']),10,0)+np.where(np.isin(season,['Ramadan','White Friday','National Day','Back to School']),12,0)+rng.normal(0,12,N),5,99)
true_price=np.clip(62+.18*(true_brand-55)+rng.normal(0,11,N),5,99); true_checkout=np.clip(true_landing+rng.normal(0,8,N),5,99); true_search=np.clip(intent_score+np.where(platform=='Google Search',8,0)+rng.normal(0,10,N),5,99); true_headline=np.clip(.55*true_creative+.45*true_fit+rng.normal(0,7,N),5,99); true_mem=np.clip(true_creative+rng.normal(0,9,N),5,99); planned_freq=np.clip(rng.normal(2.1, .45, N)+np.where(astrat=='Retargeting',.6,0),1,4.5); true_form=np.clip(true_landing+rng.normal(0,10,N),5,99); true_magnet=np.clip(true_offer+rng.normal(0,10,N),5,99); true_cycle=np.clip(68+np.where(np.isin(industry,['Real Estate','Automotive','Finance']),-18,0)+rng.normal(0,10,N),5,99); true_hook=np.clip(true_creative+rng.normal(0,9,N),5,99); true_native=np.clip(true_fit+np.where(np.isin(platform,['TikTok','Instagram','Snapchat']),8,0)+rng.normal(0,7,N),5,99); true_community=np.clip(58+.25*(true_brand-55)+rng.normal(0,12,N),5,99); true_app_rating=np.clip(rng.normal(4.15+.004*(true_brand-55),.32,N),2.6,5); true_store=np.clip(.45*true_creative+.35*true_landing+.2*true_trust+rng.normal(0,8,N),5,99); app_size=np.clip(rng.lognormal(np.log(95),.55,N),15,550); true_device=np.clip(true_mobile+rng.normal(0,7,N),5,99)
# noisy observed assessments
obs=lambda x,s=7.5: np.round(np.clip(x+rng.normal(0,s,len(x)),5,99),1)
creative=obs(true_creative); offer=obs(true_offer); landing=obs(true_landing); tracking=obs(true_tracking); mobile=obs(true_mobile); local=obs(true_local); schedule=obs(true_schedule); awareness=obs(true_brand); comp=obs(true_comp,8.5); content_fit=obs(true_fit); trend=obs(true_trend); trust=obs(true_trust)
# audience and budget adequacy
rshare=pd.Series(region).map(dict(zip(regions,rw))).to_numpy(); ashare=pd.Series(age).map({'18-24':.18,'25-34':.31,'35-44':.24,'45-54':.16,'55+':.11}).to_numpy(); breadth=pd.Series(astrat).map({'Broad':.65,'Interest':.32,'Lookalike':.20,'Retargeting':.08,'Custom':.05}).to_numpy(); audience=np.clip((GASTAT_POP*INTERNET*rshare*ashare*np.where(gender=='All',1,.5)*breadth*rng.uniform(.72,1.12,N)).astype(int),5000,8_000_000); budget_per=budget/(audience/1000); need=pd.Series(objective).map({'Sales':22,'Traffic':15,'Awareness':10,'Lead Generation':20,'Engagement':11,'App Installs':18}).to_numpy(); budget_adequacy=obs(np.clip((budget_per/need)*50,5,99),5)
# price/AOV
price=[]
pp={'Retail & E-commerce':[.28,.44,.21,.07],'Food & Restaurants':[.50,.40,.09,.01],'Beauty & Fashion':[.28,.42,.23,.07],'Real Estate':[0,.02,.23,.75],'Education':[.08,.31,.42,.19],'Healthcare':[.12,.36,.37,.15],'Travel & Hospitality':[.07,.34,.42,.17],'Automotive':[0,.05,.32,.63],'Finance':[.05,.30,.45,.20],'Telecom':[.22,.47,.26,.05],'Technology/SaaS':[.14,.42,.33,.11],'Entertainment':[.32,.46,.18,.04]}
for i in industry:
    price.append(rng.choice(['Low','Medium','High','Premium'],p=pp[i]))
price=np.array(price)
aov=pd.Series(price).map({'Low':70,'Medium':210,'High':620,'Premium':1700}).to_numpy()*pd.Series(industry).map({'Retail & E-commerce':1.,'Food & Restaurants':.55,'Beauty & Fashion':.9,'Real Estate':2.2,'Education':1.7,'Healthcare':1.45,'Travel & Hospitality':2.,'Automotive':2.5,'Finance':1.4,'Telecom':1.1,'Technology/SaaS':1.8,'Entertainment':.7}).to_numpy()*rng.lognormal(0,.16,N)
# placements/content/bid
placements={'Google Search':['Search Results','Performance Max'],'Instagram':['Feed','Stories','Reels'],'TikTok':['For You Feed','Search'],'Snapchat':['Stories','Spotlight'],'YouTube':['In-stream','Shorts','In-feed'],'X':['Timeline','Search'],'Facebook':['Feed','Stories','Reels'],'LinkedIn':['Feed','Message Ads']}; contents={'Google Search':['Text','Responsive Search'],'Instagram':['Image','Video','Carousel','Reel'],'TikTok':['Short Video','UGC Video','Creator Video'],'Snapchat':['Vertical Video','Story','AR Lens'],'YouTube':['Video','Short Video'],'X':['Text+Image','Video','Carousel'],'Facebook':['Image','Video','Carousel','Reel'],'LinkedIn':['Single Image','Video','Document','Lead Gen Form']}; placement=np.array([rng.choice(placements[p]) for p in platform]); content=np.array([rng.choice(contents[p]) for p in platform]); bidding_strategy=np.array([rng.choice({'Sales':['Maximize Conversions','Target ROAS','Target CPA'],'Traffic':['Maximize Clicks','Manual CPC'],'Awareness':['Target CPM','Maximize Reach'],'Lead Generation':['Maximize Conversions','Target CPA','Maximize Clicks'],'Engagement':['Maximize Engagement','Target CPM'],'App Installs':['Target CPA','Maximize Conversions']}[o]) for o in objective])
# outcomes use latent values, never observed scores
util=np.clip(rng.normal(.90,.06,N)-.00030*(true_comp-55)+np.where(true_tracking<40,-.025,0),.68,1); spend=budget*util; cpm=np.array([base_cpm[p] for p in platform])*np.exp((true_comp-55)/120)*np.exp(-(true_creative-60)/380)*np.exp(-(true_fit-60)/520)*rng.lognormal(0,.16,N); cpm=np.clip(cpm,6,125); impressions=np.maximum(100,(spend/cpm*1000).astype(int)); freq=np.clip(rng.normal(2,.5,N)+np.where(astrat=='Retargeting',.7,0),1.05,4.5); reach=np.minimum(impressions,(impressions/freq).astype(int))
# V5: latent execution uncertainty calibrated to include real Saudi underperformance tails.
# It is deliberately NOT exported as a feature. Most campaigns are near 1.0;
# a small negative tail creates realistic weak campaigns, while a small breakout tail
# prevents truncating the high-performing public success-story range.
execution_mult=np.ones(N)
_u=rng.random(N)
_neg=_u<.08
_pos=(_u>=.08)&(_u<.12)
execution_mult[_neg]=rng.uniform(.38,.78,_neg.sum())
execution_mult[_pos]=rng.uniform(1.18,1.60,_pos.sum())
intent_ctr=pd.Series(intent).map({'Cold':.90,'Warm':1.08,'Hot':1.22}).to_numpy(); intent_cvr=pd.Series(intent).map({'Cold':.66,'Warm':1.08,'Hot':1.52}).to_numpy(); strat_ctr=pd.Series(astrat).map({'Broad':.92,'Interest':1.,'Lookalike':1.08,'Retargeting':1.18,'Custom':1.12}).to_numpy(); strat_cvr=pd.Series(astrat).map({'Broad':.76,'Interest':.93,'Lookalike':1.08,'Retargeting':1.42,'Custom':1.22}).to_numpy(); ctr_mean=np.array([base_ctr[p] for p in platform])*fit*intent_ctr*strat_ctr*np.exp((true_creative-60)/180)*np.exp((true_offer-60)/250)*np.exp((true_fit-60)/230)*np.exp((true_local-60)/500)*np.exp((true_schedule-60)/560)*season_mult; ctr_mean*=np.where(objective=='Traffic',np.exp((true_search-60)/210)*np.exp((true_headline-60)/250),1); ctr_mean*=np.where(objective=='Engagement',np.exp((true_hook-60)/260)*np.exp((true_native-60)/280),1); ctr_mean*=np.sqrt(execution_mult); ctr_p=np.clip(ctr_mean*rng.lognormal(0,.32,N),.0008,.15); clicks=rng.binomial(impressions,ctr_p)
obj_base=pd.Series(objective).map({'Sales':.018,'Traffic':.008,'Awareness':.004,'Lead Generation':.034,'Engagement':.007,'App Installs':.026}).to_numpy(); price_factor=pd.Series(price).map({'Low':1.10,'Medium':1.,'High':.86,'Premium':.70}).to_numpy(); cvr_mean=obj_base*fit*intent_cvr*strat_cvr*np.exp((true_landing-60)/155)*np.exp((true_offer-60)/190)*np.exp((true_tracking-60)/320)*np.exp((true_brand-60)/380)*price_factor; cvr_mean*=np.where(objective=='Sales',np.exp((true_price-60)/260)*np.exp((true_checkout-60)/250)*np.exp((true_trust-60)/280),1); cvr_mean*=np.where(objective=='Lead Generation',np.exp((true_form-60)/220)*np.exp((true_magnet-60)/240)*np.exp((true_cycle-60)/300),1); cvr_mean*=np.where(objective=='App Installs',np.exp((true_store-60)/220)*np.exp((true_device-60)/250)*np.exp((true_app_rating-4)/1.8),1); cvr_mean*=execution_mult; cvr_p=np.clip(cvr_mean*rng.lognormal(0,.42,N),.0003,.20); conversions=rng.binomial(clicks,cvr_p)
er_mean=np.array([base_er[p] for p in platform])*fit*np.exp((true_creative-60)/170)*np.exp((true_fit-60)/170)*np.exp((true_trend-60)/220)*intent_ctr; er_mean*=np.where(objective=='Engagement',np.exp((true_hook-60)/180)*np.exp((true_native-60)/180)*np.exp((true_community-60)/260),1); er_mean*=np.sqrt(execution_mult); er_p=np.clip(er_mean*rng.lognormal(0,.30,N),.0008,.12); engagements=rng.binomial(impressions,er_p)
revenue=conversions*aov*rng.lognormal(0,.18,N); ctr=clicks/impressions; cpc=np.divide(spend,clicks,out=np.full(N,np.nan),where=clicks>0); cpm_actual=spend/impressions*1000; cvr=np.divide(conversions,clicks,out=np.zeros(N),where=clicks>0); cpa=np.divide(spend,conversions,out=np.full(N,np.inf),where=conversions>0); roas=revenue/spend; er=engagements/impressions
# targets
sales_base=pd.Series(industry).map({'Retail & E-commerce':2.5,'Food & Restaurants':2.0,'Beauty & Fashion':2.4,'Real Estate':2.8,'Education':2.5,'Healthcare':2.4,'Travel & Hospitality':2.7,'Automotive':2.9,'Finance':2.8,'Telecom':2.5,'Technology/SaaS':2.8,'Entertainment':2.2}).to_numpy()*1.08; traffic_base=pd.Series(platform).map({'Google Search':3.8,'Instagram':2.7,'TikTok':2.3,'Snapchat':2.2,'YouTube':3.,'X':2.8,'Facebook':2.7,'LinkedIn':5.2}).to_numpy()*.85; aware_base=pd.Series(platform).map({'Google Search':38,'Instagram':31,'TikTok':28,'Snapchat':26,'YouTube':30,'X':29,'Facebook':29,'LinkedIn':52}).to_numpy()*.90; lead_base=pd.Series(industry).map({'Retail & E-commerce':75,'Food & Restaurants':55,'Beauty & Fashion':70,'Real Estate':225,'Education':145,'Healthcare':165,'Travel & Hospitality':155,'Automotive':210,'Finance':225,'Telecom':115,'Technology/SaaS':180,'Entertainment':75}).to_numpy()*.45; install_base=pd.Series(industry).map({'Retail & E-commerce':42,'Food & Restaurants':30,'Beauty & Fashion':39,'Real Estate':65,'Education':50,'Healthcare':55,'Travel & Hospitality':47,'Automotive':64,'Finance':68,'Telecom':42,'Technology/SaaS':50,'Entertainment':33}).to_numpy()*1.00; engage_base=pd.Series(platform).map({'Google Search':.005,'Instagram':.022,'TikTok':.030,'Snapchat':.020,'YouTube':.015,'X':.017,'Facebook':.019,'LinkedIn':.012}).to_numpy()*1.25; difficulty=rng.lognormal(0,.07,N); target=np.select([objective=='Sales',objective=='Traffic',objective=='Awareness',objective=='Lead Generation',objective=='Engagement'],[sales_base*difficulty,traffic_base*difficulty,aware_base*difficulty,lead_base*difficulty,engage_base*difficulty],default=install_base*difficulty); success=np.select([objective=='Sales',objective=='Traffic',objective=='Awareness',objective=='Lead Generation',objective=='Engagement'],[roas>=target,cpc<=target,cpm_actual<=target,cpa<=target,er>=target],default=cpa<=target).astype(int); ttype=pd.Series(objective).map({'Sales':'ROAS','Traffic':'CPC_SAR','Awareness':'CPM_SAR','Lead Generation':'CPA_SAR','Engagement':'Engagement_Rate','App Installs':'CPA_SAR'}).to_numpy()

# dataframe then strictly prior history features
df=pd.DataFrame({'Campaign_ID':[f'KSA-V5-{i+1:06d}' for i in range(N)],'Brand_ID':brand_id_rows,'Start_Date':date,'Duration_Days':duration,'Budget_SAR':np.round(budget,2),'Platform':platform,'Placement':placement,'Content_Type':content,'Campaign_Objective':objective,'Industry':industry,'Region':region,'City':city,'Target_Age':age,'Target_Gender':gender,'Audience_Strategy':astrat,'Audience_Intent':intent,'Estimated_Audience_Size':audience,'Budget_Per_1000_Audience_SAR':np.round(budget_per,3),'Budget_Adequacy_Score':budget_adequacy,'Bidding_Strategy':bidding_strategy,'Brand_Maturity':maturity,'Season':season,'Creative_Quality_Score':creative,'Offer_Strength_Score':offer,'Landing_Page_Quality_Score':landing,'Tracking_Readiness_Score':tracking,'Mobile_Readiness_Score':mobile,'Arabic_Localization_Score':local,'Scheduling_Alignment_Score':schedule,'Brand_Awareness_Score':awareness,'Auction_Competition_Score':comp,'Content_Audience_Fit_Score':content_fit,'Trend_Relevance_Score':trend,'Trust_Score':trust,'Discount_Percentage':np.round(discount,1),'Product_Price_Band':price,'Expected_AOV_SAR':np.round(aov,2),'Price_Competitiveness_Score':np.where(objective=='Sales',obs(true_price),np.nan),'Checkout_Ease_Score':np.where(objective=='Sales',obs(true_checkout),np.nan),'Search_Intent_Score':np.where(objective=='Traffic',obs(true_search),np.nan),'Headline_Relevance_Score':np.where(objective=='Traffic',obs(true_headline),np.nan),'Creative_Memorability_Score':np.where(objective=='Awareness',obs(true_mem),np.nan),'Planned_Frequency':np.where(objective=='Awareness',np.round(planned_freq,2),np.nan),'Lead_Form_Ease_Score':np.where(objective=='Lead Generation',obs(true_form),np.nan),'Lead_Magnet_Strength_Score':np.where(objective=='Lead Generation',obs(true_magnet),np.nan),'Sales_Cycle_Simplicity_Score':np.where(objective=='Lead Generation',obs(true_cycle),np.nan),'Hook_Strength_Score':np.where(objective=='Engagement',obs(true_hook),np.nan),'Native_Format_Fit_Score':np.where(objective=='Engagement',obs(true_native),np.nan),'Community_Affinity_Score':np.where(objective=='Engagement',obs(true_community),np.nan),'App_Store_Rating':np.where(objective=='App Installs',np.round(true_app_rating+rng.normal(0,.12,N),2),np.nan),'App_Store_Page_Quality_Score':np.where(objective=='App Installs',obs(true_store),np.nan),'App_Size_MB':np.where(objective=='App Installs',np.round(app_size,1),np.nan),'Device_Compatibility_Score':np.where(objective=='App Installs',obs(true_device),np.nan),'Target_KPI_Type':ttype,'Target_KPI_Value':np.round(target,4),'Spend_SAR':np.round(spend,2),'Impressions':impressions,'Reach':reach,'Clicks':clicks,'Conversions':conversions,'Engagements':engagements,'CTR':np.round(ctr,6),'CPC_SAR':np.round(cpc,3),'CPM_SAR':np.round(cpm_actual,3),'Conversion_Rate':np.round(cvr,6),'CPA_SAR':np.where(np.isfinite(cpa),np.round(cpa,3),np.nan),'Revenue_SAR':np.round(revenue,2),'ROAS':np.round(roas,4),'Engagement_Rate':np.round(er,6),'Success':success})
df=df.sort_values(['Start_Date','Brand_ID','Campaign_ID']).reset_index(drop=True)
# history strictly shifted
for group, prefix in [(['Brand_ID'],'Brand'),(['Brand_ID','Platform'],'Brand_Platform'),(['Brand_ID','Campaign_Objective'],'Brand_Objective')]:
    g=df.groupby(group,sort=False)
    cnt=g.cumcount(); df[f'{prefix}_Prior_Campaigns']=cnt
    succ_prior=g['Success'].cumsum()-df['Success']; df[f'{prefix}_Prior_Success_Rate']=np.where(cnt>0,succ_prior/cnt,np.nan)
    ctr_prior=g['CTR'].cumsum()-df['CTR']; df[f'{prefix}_Prior_CTR']=np.where(cnt>0,ctr_prior/cnt,np.nan)
    cvr_prior=g['Conversion_Rate'].cumsum()-df['Conversion_Rate']; df[f'{prefix}_Prior_CVR']=np.where(cnt>0,cvr_prior/cnt,np.nan)
df['Days_Since_Last_Campaign']=df.groupby('Brand_ID')['Start_Date'].diff().dt.days
# recent 5 campaign success (prior only)
df['Brand_Recent5_Success_Rate']=df.groupby('Brand_ID')['Success'].transform(lambda s:s.shift(1).rolling(5,min_periods=1).mean())

# sealed brand/time splits
allb=np.array(sorted(df.Brand_ID.unique())); rr=np.random.default_rng(SEED+77); rr.shuffle(allb); nl=int(round(.15*len(allb))); ng=int(round(.10*len(allb))); lock=set(allb[:nl]); group=set(allb[nl:nl+ng]); dev=set(allb[nl+ng:]); lock_start=pd.Timestamp('2026-04-01')
df['Split_Role']=np.select([df.Brand_ID.isin(lock)&(df.Start_Date>=lock_start),df.Brand_ID.isin(group)&(df.Start_Date<lock_start),df.Brand_ID.isin(dev)&(df.Start_Date>=lock_start),df.Brand_ID.isin(dev)&(df.Start_Date<lock_start)],['FINAL_LOCKBOX_UNSEEN_BRANDS_V5','GROUP_VALIDATION_UNSEEN_BRANDS','TEMPORAL_HOLDOUT_SEEN_BRANDS_V5','DEVELOPMENT'],default='HISTORY_ONLY_NOT_FOR_MODEL_SELECTION')
# write
out=Path('/mnt/data'); full=out/'saudi_marketing_campaigns_v5.csv'; pre=out/'saudi_marketing_campaigns_v5_prelaunch.csv'; lockf=out/'saudi_marketing_v5_final_lockbox.csv'; brandsf=out/'saudi_marketing_v5_brands.csv'
df.to_csv(full,index=False,encoding='utf-8-sig'); brands.to_csv(brandsf,index=False,encoding='utf-8-sig')
post=['Target_KPI_Type','Target_KPI_Value','Spend_SAR','Impressions','Reach','Clicks','Conversions','Engagements','CTR','CPC_SAR','CPM_SAR','Conversion_Rate','CPA_SAR','Revenue_SAR','ROAS','Engagement_Rate']; precols=[c for c in df.columns if c not in post]; df[precols].to_csv(pre,index=False,encoding='utf-8-sig'); df[df.Split_Role=='FINAL_LOCKBOX_UNSEEN_BRANDS_2026'].to_csv(lockf,index=False,encoding='utf-8-sig')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); manifest={'seed':SEED,'dataset_version':'V5','created_before_model_training':True,
'evidence_calibration_file':'Saudi_Marketing_Evidence_Training_Ready_V5.xlsx',
'evidence_rows_total':114,'evidence_rows_v5_calibration':88,'performance_negative_evidence_rows':11,
'calibration_policy':'Published success evidence used for range/tail calibration only; performance negatives used to preserve weak tail; no public post-launch KPI is a model input.','lockbox_start':'2026-04-01','brand_counts':{'development':len(dev),'group_validation':len(group),'final_lockbox':len(lock)},'row_counts':df.Split_Role.value_counts().to_dict(),'lockbox_brand_ids_sha256':hashlib.sha256(('\n'.join(sorted(lock))).encode()).hexdigest(),'full_dataset_sha256':sha(full),'prelaunch_dataset_sha256':sha(pre),'lockbox_file_sha256':sha(lockf)}; (out/'saudi_marketing_v5_sealed_split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
# summary
audit={'dataset_version':'V5','calibration_basis':'88 real Saudi evidence rows (77 successes + 11 performance negatives; success-story publication bias explicitly not used as base-rate)', 'rows':len(df),'columns':len(df.columns),'brands':df.Brand_ID.nunique(),'date_min':str(df.Start_Date.min().date()),'date_max':str(df.Start_Date.max().date()),'success_rate':float(df.Success.mean()),'success_by_objective':df.groupby('Campaign_Objective').Success.agg(['count','mean']).round(5).to_dict('index'),'split_rows':df.Split_Role.value_counts().to_dict()}
(out/'saudi_marketing_v5_generation_summary.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
