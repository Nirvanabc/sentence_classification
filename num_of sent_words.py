dict_file = open("prepared_dict")

for pair in dict_file:
    pair = pair.split()
    d[pair[0]] = int(pair[1])
    
all_num = sent_num = 0
for sent in f.readlines()[:-3]:
    sent = sent.split()
    for word in sent:
        all_num += 1
        if word in d:
            sent_num += 1
                
