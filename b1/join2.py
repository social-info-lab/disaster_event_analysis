
import json,glob,csv,sys
from collections import Counter

csvwriter = None
csvfields = None
def print_csvrow(dct):
    global csvwriter, csvfields
    import csv
    if not csvwriter:
        csvfields = list(dct.keys())
        csvwriter = csv.DictWriter(sys.stdout, fieldnames=csvfields)
        csvfields = set(csvfields)
        csvwriter.writeheader()
    assert all(k in csvfields for k in dct.keys())
    csvwriter.writerow(dct)

emdat_kb = json.load(open("emdat-disaster-instance-info.json"))
gtd_kb1   = json.load(open("gtd-disaster-instance-info.json"))
# print(list(emdat_kb)[:5])
# print(list(gtd_kb1)[:5])
gtd_kb = {}
for eid1,value in gtd_kb1.items():
    pp = eid1.split("-")
    assert len(pp)==5
    assert pp[0]=='attack'
    eid2 = f"attack-{pp[4]}-{pp[1]}-{pp[2]}-{pp[3]}"
    # print(eid1,eid2)
    gtd_kb[eid2] = value
del gtd_kb1

assert len(  set(emdat_kb.keys()) & set(gtd_kb.keys())  )==0  

kb = dict(  list(emdat_kb.items()) + list(gtd_kb.items())  )

efiles = []
efiles += glob.glob("gtd-passing-phase-2-en-9.23/*.jsonl")
efiles += glob.glob("nd-passing-phase-2-en-9.23/*.jsonl")

kbkeys = ["num fatalities", "num injured",
        "total deaths","num injured","num affected","total affected",
        "total damage adjusted US$",]

eid_newsff = {}
for ff in efiles:
    eid = ff.split("/")[-1].replace("-finalout.jsonl","")
    eid_newsff[eid] = ff
unseen_eids = set(kb.keys()) - set(eid_newsff)
for eid in unseen_eids:
    eid_newsff[eid] = None

for (eid,ff) in eid_newsff.items():
    out = {}
    out['eid'] = eid

    if ff is None:
        out['num_articles'] = 0
    else:
        articles = [json.loads(line) for line in open(ff)]
        out['num_articles'] = len(articles)
        out['article_pub_countries'] = ' '.join(f"{k}:{c}" for (k,c) in Counter([a['country'] for a in articles]).most_common(100))

    out['lookup'] = (
            "in_gtd_json" if eid in gtd_kb else
            "in_emdat_json" if eid in emdat_kb else
            "other")
    pp = eid.split("-")
    out['e_type']    = pp[0]
    out['e_country'] = pp[1]
    out['e_date'] = '-'.join(pp[2:])
    kbinfo = kb.get(eid, {})
    for key in kbkeys:
        value = kbinfo.get(key,"")
        value = "" if value=="-999" else value
        outkey = key
        out[outkey] = value
    if out['num fatalities'] != '' and out['total deaths']=='':
        out['total deaths'] = out['num fatalities']
    del out['num fatalities']
    # print(json.dumps(out))
    print_csvrow(out)

