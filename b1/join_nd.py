# Joins EMDAT metadata along with our per-event news coverage data.
# output from this script does NOT include EMDAT instances with 0 news coverage!!

import json,glob

kb = json.load(open("emdat-disaster-instance-info.json"))
# efiles = glob.glob("passing-phase-2-en-var1/*.jsonl")
efiles = glob.glob("nd-passing-phase-2-en-9.23/*.jsonl")
for ff in efiles:
    eid = ff.split("/")[-1].replace("-finalout.jsonl","")
    # print(eid in kb, eid)
    if eid not in kb: 
        print("NOT IN KB",ff, " -- EID",eid)
        continue
    # else:
    #     print("INKB")
    #     continue

    for dd in (json.loads(line) for line in open(ff)):
        # del dd['text']

        dd['eid']=eid
        # dd['e_date_orig'] = "orig" + "-".join(eid.split("-")[-3:])
        dd['e_date'] = "-".join(eid.split("-")[-3:]).replace("-xx","-01")
        dd['e_type']=eid.split("-")[0]
        dd['e_country']=eid.split("-")[1]

        for k in ['total affected','num affected','total deaths', 'num homeless', 'num injured']:
            val = kb[eid][k] if eid in kb else ""
            val = "" if val=="-999" else val
            k2 = k.replace(" ","_").replace("num","n").replace("affected","aff").replace("total","tot").replace("homeless","hl").replace("injured","inj").replace("deaths","de")
            dd[k2] = val
        dd['a_country'] = dd.pop('country')
        dd['a_date'] = dd.pop('date')
        dd['a_text'] = dd.pop('text')

        print(json.dumps(dd))
        
