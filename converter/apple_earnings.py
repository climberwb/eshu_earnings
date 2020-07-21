import spacy
nlp = spacy.load('en_core_web_lg')

txt = " ".join(open('./apple_earnings.txt','r').readlines())
doc = nlp(txt)
sentences = [{'text':str(d), 'labels':[]} for d in doc.sents]

import pandas as pd
import json

with open('e_sentences.json', 'w+') as csvfile:
    #import pdb; pdb.set_trace()
    for s in sentences:
        csvfile.write(json.dumps(s)+"\n")
    #csvfile.write(json.dumps(sentences))
        #print(s)
        #import pdb; pdb.set_trace()
        #csvfile.write(json.dump(s))

# import csv
# with open('e_sentences.csv', 'w', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile,
#                            )
#     spamwriter.writerow(sentences)
 