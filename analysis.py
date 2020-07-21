import json
import os
import spacy 
import pandas as pd
from knn_modeling import KNNModeling,SVMModeling,RandomForestModeling,GradBoostingModeling 
import numpy as np

from nltk.stem.porter import *
porterStemmer = PorterStemmer()
## TODO try this out on larger model
nlp = spacy.load('en_core_web_lg')

LABELS = { 
    'Products&Services':0, 
    'financial':1, 
    'geo':2, 
    'contextual':3,
    'unimportant':4
}
data_lines = []
all_companies = []
present_lines = []
entity_lines = []
file_num_lines = []
company_names = [f for f in os.listdir('./data')]
print('pre')
np.random.shuffle(company_names)
for f in company_names: 
    
    with open('data/{0}'.format(f),encoding="utf-8") as json_file:
        file_lines = json_file.readlines()
        file_num_lines.append(len(file_lines))
        company_line = []
        present_tense = []
        print(f,len(file_lines))
        for line in  file_lines:
            data = json.loads(line)
            # print(data['labels'][0][2])
            document = nlp(data['text'])
            # doc_labs = [e.label_ for e in document.ents]
            

            present_count=0
            past_count = 0
            numb_card = 0
            numb_keys = 0
            numb_plur = 0
            # ents
            org_lab = 0
            mon_lab = 0
            gpe_lab = 0
            person_lab = 0
            text=""
            for token in document:
                 
                morph = nlp.vocab.morphology.tag_map[token.tag_]
                text+= " "+str(token.lemma_)
                if 'Tense_pres' in morph.keys(): 
                    present_count+=1
                elif 'Tense_past' in morph.keys(): 
                    past_count+=1
                elif 'NumType_card'in morph.keys():
                    numb_card+=1
                elif 'Number_sing' in morph.keys():
                    numb_keys+=1
                elif 'Number_plur' in morph.keys():
                    numb_plur+=1

            for ent in document.ents:
                if ent.label_ == "ORG":
                    org_lab+=1
                if ent.label_ == "MONEY":
                    mon_lab+=1
                if ent.label_ == "GPE":
                    gpe_lab+=1
                if ent.label_ == "PERSON":
                    person_lab+=1
                # print(morph.keys(), token)
            company_line.append({
                "label":LABELS[data['labels'][0][2]],
                "text":text.strip(),
                "company": f,
                "present": [present_count,past_count,numb_card,numb_plur,gpe_lab,mon_lab,person_lab]
                 })
            
            # if ('GPE' in doc_labs):
            #     entity_lines.append({
            #         "label":LABELS[data['labels'][0][2]],
            #         "text":data['text']})
            # for token in document:
            #     # 'Tense_past': True, 'VerbForm_fin'
            #     morph = nlp.vocab.morphology.tag_map[token.tag_]
            #     if 'Tense_pres' in morph.keys() and 'VerbForm_fin' in morph.keys(): 
            #         print(token,morph)
            #         present_lines.append({
            #         "label":LABELS[data['labels'][0][2]],
            #         "text":data['text']})
            #         break
        all_companies.append(company_line)   
present = [ pd.DataFrame(c)['present'] for c in all_companies ]
text = [ pd.DataFrame(c)['text'] for c in all_companies ]
labels = [ pd.DataFrame(c)['label'] for c in all_companies ]
k_model = KNNModeling((text,present), labels, n_neighbors=32, test_set_num = file_num_lines[-1])
# import pdb; pdb.set_trace()
# for i,x in enumerate(k_model.X):

preds = k_model.predict( k_model.X_ind)
for i,p in enumerate(preds):
    accuracy = sum(p == labels[i])/len(labels[i])
    print(accuracy)

rand_model = RandomForestModeling((text,present), labels, test_set_num = file_num_lines[-1])
preds = rand_model.predict( rand_model.X_ind)
for i,p in enumerate(preds):
    accuracy = sum(p == labels[i])/len(labels[i])
    print(accuracy)

# accuracy = sum(rand_model.predict( rand_model.X) == rand_model.y.values)/file_num_lines[-1]
                        
svm_model = SVMModeling((text,present), labels, test_set_num = file_num_lines[-1])
preds = svm_model.predict( rand_model.X_ind)
for i,p in enumerate(preds):
    accuracy = sum(p == labels[i])/len(labels[i])
    print(accuracy)

grad_model = GradBoostingModeling((text,present), labels, test_set_num = file_num_lines[-1])
preds = grad_model.predict( rand_model.X_ind)
for i,p in enumerate(preds):
    accuracy = sum(p == labels[i])/len(labels[i])
    print(accuracy)
# accuracy = sum(svm_model.predict( svm_model.X) == svm_model.y.values)/file_num_lines[-1]

    # print(data[0])
    # for p in data['people']:
    #     print('Name: ' + p['name'])
    #     print('Website: ' + p['website'])
    #     print('From: ' + p['from'])
    #     print('')