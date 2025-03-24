import json
import pandas as pd
import os
import statistics

def get_stats_on_data():
    d = {}
    en = os.listdir('nd-passing-phase-2-en-9.23')
    es = os.listdir('nd-passing-phase-2-es-9.23')
    fr = os.listdir('nd-passing-phase-2-fr-9.23')

    d["en"] = {}
    d["en"]["phase1"] = []
    d["en"]["phase2"] = []
    for fn in en:
        if "attack" not in fn:
            with open('nd-passing-phase-2-en-9.23/'+fn.replace('.xlsx','-finalout.jsonl'),encoding = 'utf-8') as json_file:
                json_list = list(json_file)
            f = open('9.23exps-en-nd/'+fn.replace('-finalout.jsonl','-filteredtext.json'),encoding="utf-8")
            data = json.load(f)
        else:
            with open('gtd/gtd-passing-phase-2-en-9.23/'+fn.replace('.xlsx','-finalout.jsonl'),encoding = 'utf-8') as json_file:
                json_list = list(json_file)
            f = open('gtd/en/'+fn.replace('-finalout.jsonl','-filteredtext.json'),encoding="utf-8")
            data = json.load(f)
        d["en"]['phase1'].append(len(data["arr"]))
        d["en"]['phase2'].append(len(json_list))#,round(len(json_list)/max(.0001,len(data["arr"])),4)])
    d["es"] = {}
    d["es"]["phase1"] = []
    d["es"]["phase2"] = []
    for fn in es:
        if "attack" not in fn:
            with open('nd-passing-phase-2-es-9.23/'+fn.replace('.xlsx','-finalout.jsonl'),encoding = 'utf-8') as json_file:
                json_list = list(json_file)
            f = open('9.23exps-es-nd/'+fn.replace('-finalout.jsonl','-filteredtext.json'),encoding="utf-8")
            data = json.load(f)
        else:
            with open('gtd/gtd-passing-phase-2-es-9.23/'+fn.replace('.xlsx','-finalout.jsonl'),encoding = 'utf-8') as json_file:
                json_list = list(json_file)
            f = open('gtd/es/'+fn.replace('-finalout.jsonl','-filteredtext.json'),encoding="utf-8")
            data = json.load(f)
        d["es"]['phase1'].append(len(data["arr"]))
        d["es"]['phase2'].append(len(json_list))
    d["fr"] = {}
    d["fr"]["phase1"] = []
    d["fr"]["phase2"] = []
    for fn in fr:
        if "attack" not in fn:
            with open('nd-passing-phase-2-fr-9.23/'+fn.replace('.xlsx','-finalout.jsonl'),encoding = 'utf-8') as json_file:
                json_list = list(json_file)
            f = open('9.23exps-fr-nd/'+fn.replace('-finalout.jsonl','-filteredtext.json'),encoding="utf-8")
            data = json.load(f)
        else:
            with open('gtd/gtd-passing-phase-2-fr-9.23/'+fn.replace('.xlsx','-finalout.jsonl'),encoding = 'utf-8') as json_file:
                json_list = list(json_file)
            f = open('gtd/fr/'+fn.replace('-finalout.jsonl','-filteredtext.json'),encoding="utf-8")
            data = json.load(f)
        d["fr"]['phase1'].append(len(data["arr"]))
        d["fr"]['phase2'].append(len(json_list))
   
    print(min(d['en']['phase1']),statistics.median(d['en']['phase1']),round(statistics.mean(d['en']['phase1']),4),max(d['en']['phase1']),sum(d['en']['phase1']))
    print(min(d['en']['phase2']),statistics.median(d['en']['phase2']),round(statistics.mean(d['en']['phase2']),4),max(d['en']['phase2']),sum(d['en']['phase2']))
    print(min(d['es']['phase1']),statistics.median(d['es']['phase1']),round(statistics.mean(d['es']['phase1']),4),max(d['es']['phase1']),sum(d['es']['phase1']))
    print(min(d['es']['phase2']),statistics.median(d['es']['phase2']),round(statistics.mean(d['es']['phase2']),4),max(d['es']['phase2']),sum(d['es']['phase2']))
    print(min(d['fr']['phase1']),statistics.median(d['fr']['phase1']),round(statistics.mean(d['fr']['phase1']),4),max(d['fr']['phase1']),sum(d['fr']['phase1']))
    print(min(d['fr']['phase2']),statistics.median(d['fr']['phase2']),round(statistics.mean(d['fr']['phase2']),4),max(d['fr']['phase2']),sum(d['fr']['phase2']))


def eval():
    paths = [('english-excel-files-gt','en'),('spanish-excel-files-gt','es'),('french-excel-files-gt','fr')]
    for path,lang in paths:
        arr = []
        head = os.listdir(path)
        for h in head:
            if "~$" in h:
                continue
            if "attack" not in h:
                with open('nd-passing-phase-2-'+lang+'-9.23/'+h.replace('.xlsx','-finalout.jsonl'),encoding = 'utf-8') as json_file:
                    json_list = list(json_file)
            else:
                with open('gtd/gtd-passing-phase-2-'+lang+'-9.23/'+h.replace('.xlsx','-finalout.jsonl'),encoding = 'utf-8') as json_file:
                    json_list = list(json_file)
            # Strips the newline character
            yestextsforllm = []
            for line in json_list:
                t = json.loads(line)
                #remove special characters
                yestextsforllm.append(t["text"].replace('\r\n','').replace('\n','').replace('\t','').replace('\u2018','-').replace('\u2019','\'').replace('\u201d','-').replace('\u2013','-').replace('\u201c','-').replace('\u00a0','-').replace('\u00e9','-').replace('\u2014','-').replace('\u00f1','-').replace('\u00e1','-').replace('\u00ed','-').replace('\u00f3','-').replace('_x000D_','')[:100])
            ground_truth_df = pd.read_excel(path+"/"+h,skiprows=1)
            tp = 0
            fp = 0
            fn = 0
            print(h)
            print("\n")
            for i in range(0,len(ground_truth_df)):
                if i > 150:
                    continue
                #remove special characters
                if ground_truth_df.loc[i]['Text'].replace('\r\n','').replace('\n','').replace('\t','').replace('\u2018','-').replace('\u2019','\'').replace('\u201d','-').replace('\u2013','-').replace('\u201c','-').replace('\u00a0','-').replace('\u00e9','-').replace('\u2014','-').replace('\u00f1','-').replace('\u00e1','-').replace('\u00ed','-').replace('\u00f3','-').replace('_x000D_','')[:100] in yestextsforllm:
                    llmlabel = 1
                else:
                    llmlabel = 0
                #the llm and ground truth agree
                if llmlabel == int(ground_truth_df.loc[i]['Label']):
                    if llmlabel == 1:
                        tp = tp + 1
                #the llm and ground truth do not agree
                else:
                    if llmlabel == 1:
                        fp = fp + 1
                    else:
                        fn = fn + 1
            if tp + fp > 0:
                pr = round(tp/(tp+fp),4)
            else:
                pr = "n/a"
            if tp + fn > 0:
                re = round(tp/(tp + fn),4)
            else:
                re = "n/a"
            if pr != "n/a" and re != "n/a" and pr + re > 0:
                f1 = round((2 * pr * re) / (pr + re),4)
            else:
                f1 = "n/a"
            arr.append([h.replace('.xlsx',''),tp, fp, fn, pr, re, f1])
        stats = pd.DataFrame(arr)
        print(path,stats)

eval()
        

def repstudyeval(typ):
    disaster_keywords = {"earthquake":["earthquake","quake"],"flood":["flood"],"wildfire":["fire"],"landslide":["landslide"],"avalanche":["avalanche"],"volcano":["volcano","volcanic"],"storm":["storm","tornado","tidal wave","hurricane","cyclone","typhoon"],"attack":["attack","assassin","hijack","kidnap","barricade","kill",'murder','bomb','assault','strik','ambush','blitz','raid','invasion']}
    path = 'english-excel-files-gt'
    head = os.listdir(path)
    arr = []
    for h in head:
        if "~$" in h:
            continue
        # Strips the newline character
        ind1 = h.find('-')
        ind2 = h[ind1+1:].find('-') + ind1 + 1
        country = h[ind1+1:ind2].replace('russian federation','russia')
        country = country[0] + country[1:]
        disaster = h[:ind1]
        ground_truth_df = pd.read_excel(path+"/"+h,skiprows=1)
        match = []
        tp = 0
        fp = 0
        fn = 0
        print(h)
        for i in range(0,len(ground_truth_df)):
            if i > 150:
                continue
            index = ground_truth_df.loc[i]['Text'].find('. ')
            if typ == "title":
                text = ground_truth_df.loc[i]['Text'][:index].lower()
            else:
                text = ground_truth_df.loc[i]['Text'].lower()
            baselinelabel = 0
            if country in text:
                for dk in disaster_keywords[disaster]:
                    if dk in text:
                        baselinelabel = 1
            if baselinelabel == int(ground_truth_df.loc[i]['Label']):
                if baselinelabel == 1:
                    tp = tp + 1
            else:
                match.append(0)
                if baselinelabel == 1:
                    fp = fp + 1
                else:
                    fn = fn + 1
        if tp + fp > 0:
            pr = round(tp/(tp+fp),4)
        else:
            pr = "n/a"
        if tp + fn > 0:
            re = round(tp/(tp + fn),4)
        else:
            re = "n/a"
        if pr != "n/a" and re != "n/a" and pr + re > 0:
            f1 = round((2 * pr * re) / (pr + re),4)
        else:
            f1 = "n/a"
        arr.append([h.replace('.xlsx',''),tp, fp, fn, pr, re, f1])
    stats = pd.DataFrame(arr)
    print(stats)

