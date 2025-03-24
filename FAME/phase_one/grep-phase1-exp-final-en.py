import os
import json
from collections import Counter
filenames = ['2020-01-01.json','2020-01-02.json','2020-01-03.json','2020-01-04.json','2020-01-05.json','2020-01-06.json','2020-01-07.json','2020-01-08.json','2020-01-09.json','2020-01-10.json','2020-01-11.json','2020-01-12.json','2020-01-13.json','2020-01-14.json','2020-01-15.json','2020-01-16.json','2020-01-17.json','2020-01-18.json','2020-01-19.json','2020-01-20.json','2020-01-21.json','2020-01-22.json','2020-01-23.json','2020-01-24.json','2020-01-25.json','2020-01-26.json','2020-01-27.json','2020-01-28.json','2020-01-29.json','2020-01-30.json','2020-01-31.json','2020-02-01.json','2020-02-02.json','2020-02-03.json','2020-02-04.json','2020-02-05.json','2020-02-06.json','2020-02-07.json','2020-02-08.json','2020-02-09.json','2020-02-10.json','2020-02-11.json','2020-02-12.json','2020-02-13.json','2020-02-14.json','2020-02-15.json','2020-02-16.json','2020-02-17.json','2020-02-18.json','2020-02-19.json','2020-02-20.json','2020-02-21.json','2020-02-22.json','2020-02-23.json','2020-02-24.json','2020-02-25.json','2020-02-26.json','2020-02-27.json','2020-02-28.json','2020-02-29.json','2020-03-01.json','2020-03-02.json','2020-03-03.json','2020-03-04.json','2020-03-05.json','2020-03-06.json','2020-03-07.json','2020-03-08.json','2020-03-09.json','2020-03-10.json','2020-03-11.json','2020-03-12.json','2020-03-13.json','2020-03-14.json','2020-03-15.json','2020-03-16.json','2020-03-17.json','2020-03-18.json','2020-03-19.json','2020-03-20.json','2020-03-21.json','2020-03-22.json','2020-03-23.json','2020-03-24.json','2020-03-25.json','2020-03-26.json','2020-03-27.json','2020-03-28.json','2020-03-29.json','2020-03-30.json','2020-03-31.json','2020-04-01.json','2020-04-02.json','2020-04-03.json','2020-04-04.json','2020-04-05.json','2020-04-06.json','2020-04-07.json','2020-04-08.json','2020-04-09.json','2020-04-10.json','2020-04-11.json','2020-04-12.json','2020-04-13.json','2020-04-14.json','2020-04-15.json','2020-04-16.json','2020-04-17.json','2020-04-18.json','2020-04-19.json','2020-04-20.json','2020-04-21.json','2020-04-22.json','2020-04-23.json','2020-04-24.json','2020-04-25.json','2020-04-26.json','2020-04-27.json','2020-04-28.json','2020-04-29.json','2020-04-30.json','2020-05-01.json','2020-05-02.json','2020-05-03.json','2020-05-04.json','2020-05-05.json','2020-05-06.json','2020-05-07.json','2020-05-08.json','2020-05-09.json','2020-05-10.json','2020-05-11.json','2020-05-12.json','2020-05-13.json','2020-05-14.json','2020-05-15.json','2020-05-16.json','2020-05-17.json','2020-05-18.json','2020-05-19.json','2020-05-20.json','2020-05-21.json','2020-05-22.json','2020-05-23.json','2020-05-24.json','2020-05-25.json','2020-05-26.json','2020-05-27.json','2020-05-28.json','2020-05-29.json','2020-05-30.json','2020-05-31.json','2020-06-01.json','2020-06-02.json','2020-06-03.json','2020-06-04.json','2020-06-05.json','2020-06-06.json','2020-06-07.json','2020-06-08.json','2020-06-09.json','2020-06-10.json','2020-06-11.json','2020-06-12.json','2020-06-13.json','2020-06-14.json','2020-06-15.json','2020-06-16.json','2020-06-17.json','2020-06-18.json','2020-06-19.json','2020-06-20.json','2020-06-21.json','2020-06-22.json','2020-06-23.json','2020-06-24.json','2020-06-25.json','2020-06-26.json','2020-06-27.json','2020-06-28.json','2020-06-29.json']
from nltk.tokenize import WhitespaceTokenizer


import sys
args = sys.argv
date = ""
if(len(args) >= 3):
    keywordrep = str(args[1])
    countrycode = str(args[2])
    date = str(args[3])
elif (len(args) == 2):
    keywordrep = str(args[1])
    countrycode = str(args[2])
elif (len(args) == 1):
    keywordrep = str(args[1])
    countrycode = "none"
else:
    keywordrep = "none"

newfilenames = []
if date != "":
    if "xx" in date:
        findmonth = date[4:8]
        for f in filenames:
            if findmonth in f:
                newfilenames.append(f)
    else:
        ind = filenames.index(date)
        windowsize = 7
        if keywordrep.lower() == "volcano":
            windowsize = 12
        for i in range(0,windowsize):
            newfilenames.append(filenames[ind+i])
filenames = newfilenames


mapping = {"locust":["locust"],"flood":["flood","flooding"],"earthquake":["earthquake","quake","seismic"],"volcano":["volcano","volcanoes","volcanic"],"storm":["storm","stormed","blizzard","hurricane","thunderstorm","tornado","hailstorm","cyclone","snowstorm","typhoon","windstorm","rainstorm"],"avalanche":['avalanche'],"landslide":["mudslide","landslide","mass movement","rock fall","debris flow"],"wildfire":["wildfire","forest fire","firestorm","bushfire"]}



def load_country_dict():
    f = open("country-igos-update-more.txt", "r")
    Lines = f.readlines()
    count = 0
    country_dict={}
    k = ""
    arr = []
    for line in Lines:
        if ("===" in line):
            if arr != []:
                country_dict[k] = arr
            arr = []
        else:
            line=line.replace("\n","").replace("_"," ")
            nltk_tokens = line.split("\t")
            k = nltk_tokens[0]
            arr.append(nltk_tokens[1].lower())
    return country_dict


import subprocess
def count_keywords_noperl(filenames,kws,locations):
    print("count articles with at least one keyword in "+str(kws)+" and at least one location in "+str(locations),flush=True)
    countarr = []
    arr = []
    for fn in filenames:
        print(fn,flush=True)
        fullpath = "/mnt/nfs/work1/grabowicz/xchen4/mediacloud_temp/scott/ner/en/"+fn
        kwstring = ""
        for kw in kws:
            kwstring = kwstring + " -e '\\<"+kw.replace(" ","[ ]").replace("(","").replace(")","").replace("'","")+"\\>'"
            kwstring = kwstring + " -e '\\<"+kw.replace(" ","[ ]").replace("(","").replace(")","").replace("'","")+"s\\>'"
        locstring = ""
        for loc in locations:
            locstring = locstring + " -e '\\<"+loc.replace(" ","[ ]").replace("(","").replace(")","").replace("'","")+"\\>'"
            locstring = locstring + " -e '\\<"+loc.replace(" ","[ ]").replace("(","").replace(")","").replace("'","")+"s\\>'"
        command = ""
        if kwstring.count('-e') == 0 and locstring.count('-e') == 0:
            print("nothing")
            continue
        if kwstring.count('-e') == 0:
            command = "grep -i" + locstring + " " + fullpath
        if locstring.count('-e') == 0:
            command = "grep -i" + kwstring + " " + fullpath
        if command == "":
            command = "grep -i" + kwstring + " " + fullpath + " | grep -i" + locstring
        print(command)
        it = 0
        while it < 1:
            try:
                output = subprocess.check_output(command,shell=True)
                break
            except:
                output = -1
            it = it + 1
        if output == -1:
            op = []
        else:
            output=output.decode("utf-8")
            op = output.split("\n")
        count = 0
        for jstring in op:
            if jstring == "":
                continue
            count = count + 1
            j = json.loads(jstring)
            arr.append({"title":j["title"],"article":j["story_text"][:1200],"media name":j["media_name"],"media id":j["media_id"],"media url":j["media_url"],"date":j["publish_date"][:10],"id":j["processed_stories_id"]})
    with open('/mnt/nfs/work1/brenocon/ecai/mediacloud/exps/'+kws[0].lower()+'-'+locations[0]+'-'+date[:10]+'.json','w') as f:
        json.dump({"arr":arr},f)
    print(countarr)
    if len(locations)>0:
        print({str(kws)+str(locations[0]):countarr})
    else:
        print({str(kws):countarr})

def build_inputs_and_run(filenames,keywordrep,countrycode):
    if keywordrep == "none":
        print("no args entered")
        return()
    keywordrep = keywordrep.replace("-"," ")
    countrydict = load_country_dict()
    if countrycode not in countrydict:
        print("the country code is not in the country dictionary")
        locations = []
    else:
        locations = countrydict[countrycode]
    count_keywords_noperl(filenames,mapping[keywordrep.lower()],locations)

build_inputs_and_run(filenames,keywordrep,countrycode)
