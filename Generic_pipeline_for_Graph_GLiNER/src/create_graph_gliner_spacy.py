import json
from tqdm import tqdm
import pickle as pkl
import argparse
import networkx as nx
# python -m spacy download en_core_web_sm

from gliner import GLiNER
import spacy



def create_graph(dataset,model_name,labels,threshold):
    data_file = json.load(open('../data/{}_corpus.json'.format(dataset), 'r'))
    model=GLiNER.from_pretrained(model_name)
    if dataset=='hotpotqa':
        new_data=[]
        for i in data_file:
            d={}
            d['title']=i
            text=''.join(data_file[i])
            d['text']=text
            new_data.append(d)
        data_file=new_data
    
    for i in tqdm(range(len(data_file)),total=len(data_file)):
        sent=data_file[i]['text']
        sent=sent.split('.')
        temp=[]
        for text in sent:
            entities=model.predict_entities(text,labels=labels,threshold=threshold)
            for entity in entities:
                temp.append(entity['text'])
        data_file[i]['extracted_entities']=temp
    with open('../output/{}_corpus_entities_model.json'.format(dataset), 'w') as json_file:
        json.dump(data_file, json_file, indent=4)

    print('data_file saved to output/{}_corpus_entities_model.json'.format(dataset))
    chunks2ids={}
    for i,d in enumerate(data_file):
        chunks2ids[d['text']]=i
    
    id2kws={}
    for i,d in enumerate(data_file):
        id2kws[i]=d['extracted_entities']
    
    G=nx.Graph()
    for c,i in chunks2ids.items():
        G.add_node(i) 
    
    print('nodes created in the Graph')

    for i in range(len(data_file)):
        for j in range(i+1,len(data_file)):
            entity1=id2kws[i]
            entity2=id2kws[i]
            flag=0
            for entity in entity1:
                if entity in entity2:
                    flag=1
            if flag==1:
                G.add_edge(i,j)
    
    print('edges created in the Graph')

    pkl.dump(G,open('../graph/{}_corpus_entities_model_graph.pkl'.format(dataset),'wb'))
    
    print('Graph is saved to graph folder')


def create_graph_spacy(dataset):
    data_file = json.load(open('../data/{}_corpus.json'.format(dataset), 'r'))
    if dataset=='hotpotqa':
        new_data=[]
        for i in data_file:
            d={}
            d['title']=i
            text=''.join(data_file[i])
            d['text']=text
            new_data.append(d)
        data_file=new_data
    
    nlp=spacy.load('en_core_web_lg')

    for i in tqdm(range(len(data_file)),total=len(data_file)):
        sent=data_file[i]['text']
        sent=sent.split('.')
        temp=[]
        for text in sent:
            entities=nlp(text)
            for entity in entities.ents:
                if entity.root.pos_ in ['NOUN', 'PROPN']:
                    temp.append(entity.text)
        data_file[i]['extracted_entities']=temp
    with open('../output/{}_corpus_entities_model_using_spacy.json'.format(dataset), 'w') as json_file:
        json.dump(data_file, json_file, indent=4)
    
    print('data_file saved to output/{}_corpus_entities_model_using_spacy.json'.format(dataset))

    chunks2ids={}
    for i,d in enumerate(data_file):
        chunks2ids[d['text']]=i
    
    id2kws={}
    for i,d in enumerate(data_file):
        id2kws[i]=d['extracted_entities']
    
    G=nx.Graph()
    for c,i in chunks2ids.items():
        G.add_node(i) 
    
    print('nodes created in the Graph')

    for i in range(len(data_file)):
        for j in range(i+1,len(data_file)):
            entity1=id2kws[i]
            entity2=id2kws[i]
            flag=0
            for entity in entity1:
                if entity in entity2:
                    flag=1
            if flag==1:
                G.add_edge(i,j)
    
    print('edges created in the Graph')

    pkl.dump(G,open('../graph/{}_corpus_entities_model_graph_using_spacy.pkl'.format(dataset),'wb'))
    
    print('Graph is saved to graph folder')

    

    
if __name__ == '__main__':
    # Get the first argument

    # Define arguments with argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',type=str,help="choose gliner or spacy")

    # Add dataset argument
    parser.add_argument('--dataset', type=str, help="Path to the dataset")

    # Add model_name argument with a default value
    parser.add_argument('--model_name', type=str, default="urchade/gliner_large-v2.1", help="Model name to use, default is 'urchade/gliner_large-v2.1'")

    # Add labels argument with choices and a default value
    parser.add_argument('--labels', type=str, choices=["default", "custom"], default="default", help="Label set, default is 'default'")

    # Add threshold argument with a default value of 0.5
    parser.add_argument('--threshold', type=float, default=0.5, help="Confidence threshold for predictions, default is 0.5")

    # Parse the arguments
    args = parser.parse_args()

    # Access dataset, model_name, labels, and threshold directly from parsed arguments
    model=args.model

    dataset = args.dataset
    
    if model.lower()=='gliner':
        model_name = args.model_name

        # Handle labels based on user input (default is already set in parser)
        if args.labels == "default":
            labels = ['cardinal', 'year', 'event', 'facilities', 'gpe', 'language', 'law', 'location', 'money',
                    'nationality', 'religious', 'political', 'ordinal', 'organization', 'percent', 'person',
                    'product', 'quantity', 'time', 'WORK_OF_ART']
        else:
            labels = input('Please enter custom labels separated by space: ').split(' ')

        # Access threshold directly
        threshold = args.threshold

        # Example print statements to check the values
        print(f"Dataset: {dataset}")
        print(f"Model Name: {model_name}")
        print(f"Labels: {labels}")
        print(f"Threshold: {threshold}")
    
    elif model.lower()=='spacy':
        create_graph_spacy(dataset)

    # elif model.lower()=='spacy':
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--dataset', type=str)

    #     args = parser.parse_args()
    #     dataset = args.dataset

    #     create_graph_spacy(dataset)
