
import numpy as np 
import json
import time
import openai
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def dispatch_openai_requests(
    messages_list: list[list[dict[str,str]]],
    model: str,
    temperature: float
) -> list[str]:

    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


openai.organization = "[ORG NAME]"
openai.api_key = "[KEY]"


evaldisasternames = ['avalanche-afghanistan-2020-02-13','avalanche-turkey-2020-02-04','earthquake-china-2020-01-19','earthquake-china-2020-02-06','earthquake-china-2020-05-18','earthquake-croatia-2020-03-22','earthquake-greece-2020-05-15','earthquake-honduras-2020-05-02','earthquake-indonesia-2020-01-07','earthquake-indonesia-2020-01-18','earthquake-indonesia-2020-01-19','earthquake-indonesia-2020-01-27','earthquake-indonesia-2020-02-05','earthquake-indonesia-2020-02-08','earthquake-indonesia-2020-03-18','earthquake-indonesia-2020-03-31','earthquake-indonesia-2020-04-14','earthquake-indonesia-2020-04-25','earthquake-iran-2020-02-23','earthquake-jamaica-2020-03-06','earthquake-japan-2020-05-06','earthquake-japan-2020-05-07','earthquake-mexico-2020-05-31','earthquake-papua new guinea-2020-01-07','earthquake-papua new guinea-2020-01-28','earthquake-papua new guinea-2020-03-22','earthquake-papua new guinea-2020-05-12','earthquake-papua new guinea-2020-05-22','earthquake-philippines-2020-03-18','earthquake-philippines-2020-04-18','earthquake-russian federation-2020-01-09','earthquake-russian federation-2020-01-29','earthquake-russian federation-2020-03-25','earthquake-russian federation-2020-04-16','earthquake-solomon islands-2020-02-26','earthquake-solomon islands-2020-03-17','earthquake-solomon islands-2020-05-27','earthquake-tonga-2020-04-05','earthquake-turkey-2020-01-24','earthquake-turkey-2020-02-13','earthquake-turkey-2020-02-23','earthquake-vanuatu-2020-04-12','flood-afghanistan-2020-03-28','flood-afghanistan-2020-04-13','flood-afghanistan-2020-05-02','flood-angola-2020-03-16','flood-angola-2020-04-18','flood-argentina-2020-02-11','flood-australia-2020-02-04','flood-bolivia-2020-02-08','flood-bolivia-2020-02-13','flood-brazil-2020-01-17','flood-brazil-2020-02-09','flood-brazil-2020-02-29','flood-brazil-2020-03-02','flood-burkina faso-2020-04-19','flood-burundi-2020-01-28','flood-burundi-2020-04-13','flood-canada-2020-04-26','flood-chad-2020-04-20','flood-china-2020-05-21','flood-colombia-2020-02-26','flood-congo-2020-03-30','flood-democratic republic of the congo-2020-03-17','flood-democratic republic of the congo-2020-04-16','flood-democratic republic of the congo-2020-05-05','flood-djibouti-2020-04-20','flood-ecuador-2020-02-22','flood-egypt-2020-03-11','flood-ethiopia-2020-04-20','flood-france-2020-05-11','flood-guatemala-2020-05-09','flood-honduras-2020-02-28','flood-india-2020-05-10','flood-india-2020-05-24','flood-indonesia-2020-01-14','flood-indonesia-2020-01-23','flood-indonesia-2020-01-28','flood-indonesia-2020-02-11','flood-indonesia-2020-02-13','flood-indonesia-2020-02-26','flood-indonesia-2020-03-04','flood-indonesia-2020-03-20','flood-indonesia-2020-03-29','flood-indonesia-2020-04-04','flood-indonesia-2020-04-09','flood-indonesia-2020-04-20','flood-indonesia-2020-04-30','flood-indonesia-2020-05-08','flood-indonesia-2020-05-22','flood-iran-2020-01-09','flood-iran-2020-02-24','flood-iran-2020-04-10','flood-iraq-2020-03-18','flood-israel-2020-01-04','flood-kazakhstan-2020-05-01','flood-kenya-2020-04-18','flood-madagaskar-2020-01-01','flood-malawi-2020-02-04','flood-malaysia-2020-04-25','flood-mozambique-2020-02-11','flood-namibia-2020-03-06','flood-new zealand-2020-02-01','flood-pakistan-2020-03-04','flood-pakistan-2020-03-31','flood-papua new guinea-2020-02-05','flood-papua new guinea-2020-03-21','flood-papua new guinea-2020-04-04','flood-peru-2020-01-23','flood-peru-2020-02-17','flood-peru-2020-03-28','flood-republic of south africa-2020-02-07','flood-rwanda-2020-02-02','flood-rwanda-2020-03-02','flood-rwanda-2020-04-17','flood-rwanda-2020-05-01','flood-somalia-2020-04-20','flood-tajikistan-2020-05-14','flood-uganda-2020-04-17','flood-uganda-2020-05-08','flood-uganda-2020-05-21','flood-united kingdom-2020-02-28','flood-united republic of tanzania-2020-01-27','flood-united republic of tanzania-2020-03-10','flood-united republic of tanzania-2020-04-22','flood-uzbekistan-2020-05-01','flood-yemen-2020-03-25','flood-yemen-2020-04-15','flood-zambia-2020-01-xx','flood-zambia-2020-03-20','landslide-ethiopia-2020-05-09','landslide-ethiopia-2020-05-28','landslide-india-2020-05-27','landslide-papua new guinea-2020-04-10','landslide-peru-2020-02-23','storm-afghanistan-2020-01-12','storm-argentina-2020-01-15','storm-australia-2020-01-19','storm-bangladesh-2020-05-20','storm-belgium-2020-02-09','storm-burundi-2020-03-16','storm-china-2020-05-01','storm-cuba-2020-05-25','storm-czech republic-2020-02-07','storm-el salvador-2020-05-31','storm-fiji-2020-01-17','storm-fiji-2020-04-08','storm-france-2020-01-21','storm-france-2020-02-08','storm-germany-2020-02-09','storm-guatemala-2020-05-30','storm-honduras-2020-05-30','storm-india-2020-01-11','storm-india-2020-05-20','storm-italy-2020-02-08','storm-madagaskar-2020-03-13','storm-mongolia-2020-01-xx','storm-oman-2020-05-27','storm-pakistan-2020-01-11','storm-philippines-2020-05-15','storm-poland-2020-02-08','storm-slovenia-2020-02-07','storm-solomon islands-2020-04-02','storm-spain-2020-01-19','storm-sri lanka-2020-05-17','storm-sweden-2020-02-07','storm-thailand-2020-04-29','storm-tonga-2020-04-06','storm-turkey-2020-01-07','storm-tuvalu-2020-01-18','storm-united kingdom-2020-02-08','storm-vanuatu-2020-04-06','storm-vietnam-2020-03-02','storm-vietnam-2020-04-22','volcano-philippines-2020-01-12','wildfire-bolivia-2020-01-xx','wildfire-china-2020-03-30','storm-united states of america-2020-01-10','storm-united states of america-2020-03-02','storm-united states of america-2020-04-06','storm-united states of america-2020-04-10','storm-united states of america-2020-05-02','flood-united states of america-2020-02-10','flood-united states of america-2020-05-17']



for dis_inst in evaldisasternames:
    arr_of_texts = []
    f = open('9.23exps-en-nd/'+dis_inst+'-filteredtext.json')
    data = json.load(f)
    for t in data["arr"]:
        arr_of_texts.append(t)
    f_record = open('qa-in-out-en-var1/'+dis_inst+'-qa-RECORD.jsonl', 'w')
    seen_and_yes = []
    seen_and_no = []
    final = []
    i = 0
    batch = []
    batchext = []
    dis_inst = dis_inst.replace('cyclone','storm')
    ind1 = dis_inst.find('-')
    ind2 = dis_inst[ind1+1:].find('-') + ind1 + 1
    country = dis_inst[ind1+1:ind2]
    country = country[0].upper() + country[1:]
    disaster = dis_inst[:ind1]
    for t in arr_of_texts: 
        skip = False
        #check if this has been seen and does not discuss the event
        for s in seen_and_no:
            if t[0][:100] == s[:100]:
                skip = True
                break
        if skip == True:
            i = i + 1
            continue
        #check if this has been seen and does discuss the event
        for s in seen_and_yes:
            if t[0][:100] == s[:100]:
                skip = True
                final.append(t)
                break
        if skip == True:
            i = i + 1
            continue
        if (i % 14 == 0 and i != 0) or i == len(arr_of_texts)-1:
            batch.append([{"role": "user", "content":t[0]+"\nDoes the text discuss a recent "+disaster+" in "+country+"?"}])
            batchext.append(t)
            while True: 
                try: 
                    completions = asyncio.run(dispatch_openai_requests(messages_list=batch,model="gpt-3.5-turbo",temperature=0))
                    break
                except:
                    print(f"openai error, redo")
                    time.sleep(10)
            index = 0
            for response in completions:
                generated_text = response.choices[0]["message"]["content"].replace("\n","")
                if "yes" in generated_text.lower():
                    val = 1
                    seen_and_yes.append(batchext[index][0])
                    final.append(batchext[index])
                else:
                    val = 0
                    seen_and_no.append(batchext[index][0])
                d_record = {"prompt": str(batch[index]).replace("[{'role': 'user', 'content': ","").replace("}]",""), "bool": val, "generated_text": generated_text}
                print(d_record)
                json.dump(d_record, f_record) 
                f_record.write('\n')
                f_record.flush()
                index = index + 1
            batch = []
            batchext = []
        else:
            batchext.append(t)
            batch.append([{"role": "user", "content":t[0]+"\nDoes the text discuss a recent "+disaster+" in "+country+"?"}])
        i = i +1

    with open('passing-phase-2-en-9.23/'+dis_inst+'-finalout.jsonl', 'w') as outfile:
        for entry in final:
            json.dump({"text":entry[0],"date":entry[1],"country":entry[2]}, outfile)
            outfile.write('\n')
