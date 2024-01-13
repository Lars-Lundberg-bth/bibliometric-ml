import pybliometrics
import copy

class Dictentry:
    def __init__(self):
        self.kw_set = set() # create empty set
        self.age_counter = 0
        self.citation_count = 0
    def add_to_set(self, new_eid, nr_refs = 0):
        if not new_eid in self.kw_set:
            self.kw_set.add(new_eid)
            self.citation_count += nr_refs
    def clear_set(self):
        self.kw_set = set()
    def get_count(self):
        return len(self.kw_set)
    def clear_set(self):
        self.kw_set = set() # set existing set to empty set
    def make_union(self, added_set):
        self.kw_set = self.kw_set.union(added_set)
    def get_set(self):
        return self.kw_set
    def add_age_counter(self,value): # age_counter is used for many purposes, not only to handle the age_factor
        self.age_counter += value
    def get_age_counter(self):
        return self.age_counter
    def get_citations(self):
        return self.citation_count

age_factor = 1.5
max_key_word_length = 5
minimum_kw_occaision = 30

years = (2013,2014,2015,2016,2017,2018,2019,2020,2021,2022) # (2012,2013,2014,2015,2016,2017,2018,2019,2020,2021)
first_part_of_search_str = 'TITLE-ABS-KEY ( {machine learning} ) AND (PUBYEAR = '

blacklist = [
    'machine learning', 'ml', 'machine learning ml', 
    'artificial intelligence', 'ai', 'artificial intelligence ai', 
    'classification', 'prediction', 'modeling', 'algorithms', 
    'prognosis', 'accuracy', 'optimization', 'performance',
    'machine-learning', 'detection', 'learning', 'machine', 
    'and machine learning', 'data', 'analysis',
    'machine learning techniques', 'machine learning methods', 
    'network', 'machine learning models', 'networks', 'model', 
    'models', 'machine learning approach', 'classification',
    'machine learning algorithms', 'machine learning algorithm', 
    'learning algorithms'
    ] 

thesaurus = [ #202
    # thesaurus defines research directions
    [['feature selection', 'feature extraction', 'principal component analysis',
      'dimensionality reduction', 'data augmentation', 'feature engineering',
      'pca', 'representation learning', 'imbalanced data', 'data fusion',
      'class imbalance', 'smote'],
     ['Data processing']], #12
    
    [['reinforcement learning', 
      'supervised learning', 
      'unsupervised learning', 'deep reinforcement learning', 
      'transfer learning', 'semi-supervised learning', 'federated learning',
      'supervised machine learning', 'active learning', 
      'unsupervised machine learning', 'explainable ai', 'interpretability',
      'adversarial machine learning', 'online learning', 'domain adaptation',
      'interpretable machine learning', 'explainable artificial intelligence',
      'q-learning', 'explainability'], 
     ['Learning paradigms']], #19

    [['cloud computing', 'gpu', 'tensorflow', 
       'smartphone', 'fpga', 'android', 'python', 
       'cloud', 'fog computing', 'mapreduce', 'spark', 'edge computing',
       'energy efficiency', '5g'], 
     ['System and hardware']], #14

    # Algorithms

    [['random forest', 'random forests', 'random forests', 'adaboost',
       'ensemble learning', 'xgboost', 'ensemble', 'gradient boosting',
       'bagging', 'boosting'],
     ['Ensemble']],  #10

    [['logistic regression', 'linear regression'],
     ['Linear']], #2

    [['knn', 'k-nn','k-nearest neighbor', 'k-nearest neighbors',],
     ['Nearest neighbor']], #4

    [['support vector machine', 'svm', 'support vector machines', 
       'support vector machine svm', 'support vector regression',
       'support vector machines', 'support vector'],
     ['Kernel based']], #7

     [['decision tree', 'decision trees'],
      ['Tree based']], #2

     [['clustering', 'k-means', 'k-means clustering',
       'cluster analysis'],
      ['Clustering']], #4

     [['naive bayes', 'naïve bayes', 'gaussian process regression',
       'bayesian optimization'],
      ['Statistical']], #4

    [['genetic algorithm', 'particle swarm opimization',
        'fuzzy logic', 'genetic programming'],
     ['Other algorithms']], #4

     [['convolutional neural network', 
       'convolutional neural networks', 'cnn', 
       'convolutional neural network cnn',
       'convolution neural network'],
      ['CNN']], #5

     [['lstm', 'recurrent neural network', 'rnn', 'long short-term memory',
       'recurrent neural networks'], 
      ['RNN']], #5

      [['neural networks', 'neural network','artificial neural network', 
        'artificial neural networks', 'ann', 'artificial neural network ann', 
        'deep learning', 'deep neural networks', 'deep neural network', 
        'deep learning dl', 'extreme learning machine', 'autoencoder',
        'generative adversarial networks', 'attention mechanism', 
        'multilayer perceptron', 'generative adversarial network'],
       ['Other neural network algorithms']], #16

     # Applications
     [['internet of things', 'iot', 'internet of things iot', 'sensors',
       'smart grid', 'wearable sensors'],
     ['Internet of things']], #6

     [['natural language processing', 'nlp', 'sentiment analysis',
       'text classification', 'opinion mining', 'bert', 
       'topic modeling', 'word embedding'],
      ['Natural language processing']], #8

     [['image processing', 'image classification', 
       'computer vision', 'image recognition', 
       'machine vision', 'image analysis', 'segmentation',
       'object detection', 'image segmentation', 'face recognition',
       'semantic segmentation'],
      ['Image processing and computer vision']], #11

     [['anomaly detection', 'security', 'malware',
       'intrusion detection', 'privacy', 'cybersecurity',
       'malware detection', 'intrusion detection system',
       'network security', 'blockchain', 'cyber security',
       'risk assessment', 'fraud detection'],
      ['Security']], #13

     [['social media', 'twitter', 'social networks',
       'fake news'],
      ['Social media']], #4

     [['covid-19', 'healthcare', 'alzheimer s disease', 
       'alzheimer’s disease', 'radiomics', 'breast cancer', 
       'bioinformatics', 'cancer', 'precision medicine', 'eeg', 
       'biomarkers', 'lung cancer', 'stroke', 'mental health', 
       'parkinson s disease', 'sars-cov-2', 'biomarker',
       'diabetes', 'electronic health records', 
       'magnetic resonance imaging', 'mri', 'activity recognition',
       'depression', 'emotion recognition', 'drug discovery',
       'human activity recognition', 'epilepsy', 
       'electroencephalography', 'computer-aided diagnosis',
       'computed tomography', 'affective computing',
       'gene expression', 'condition monitoring', 
       'medical imaging', 'ecg', 'neuroimaging',
       'schizophrenia', 'heart disease'],
      ['Healthcare']], #38

     [['data mining', 'big data', 'data science', 'text mining',
       'big data analytics', 'data analytics', 'pattern recognition', 
       'data analysis', 'information retrieval', 
       'information extraction'],
      ['Data mining']], #10

      [['fault diagnosis', 'industry 4 0', 'predictive maintenance',
        'fault detection', 'agriculture', 'climate change', 
        'robotics', 'virtual reality', 'uav', 'recommender systems',
        'decision making', 'digital twin', 'gis', 'smart city'],
      ['Other applications']] #14
    ]

thesaurus2 = [
    # thesaurus2 clusters research directions into larger research direction
     # [['Data mining', 'Healthcare', 'Social media', 'Security',
     #   'Image processing and computer vision',
     #   'Natural language processing','Internet of things', 
     #   'Other applications'],
     #  ['APPLICATIONS']]

     # [['RNN', 'CNN', 'Other neural network algorithms'],
     #  ['Neural network algorithms']],

     # [['Ensemble', 'Linear', 'Nearest neighbor', 'Kernel based',
     #   'Tree based', 'Clustering', 'Statistical', 
     #  'Other algorithms'],
     #  ['Other algorithms than neural networks']]
    
     # [['Neural network algorithms', 
     #   'Other algorithms than neural networks'],
     # ['ALGORITHMS']]
    
    ]

blackset = set()
for black_kw in blacklist:
    blackset.add(tuple(black_kw.split()))
blackset = frozenset(blackset)

syn_dict = {}
for synonym in thesaurus:
    print(synonym[1][0])
    composed_kw = tuple(synonym[1][0].split())
    print('Composed kw: ', composed_kw)
    for old_kw in synonym[0]:
        syn_dict[tuple(old_kw.split())] = composed_kw
        print('Old kw: ', tuple(old_kw.split()))
for synonym in thesaurus2:
    print(synonym[1][0])
    composed_kw = tuple(synonym[1][0].split())
    print('Composed kw: ', composed_kw)
    for old_kw in synonym[0]:
        for syn in syn_dict:
            if syn_dict[syn] == tuple(old_kw.split()):
                print('syn:', syn,' old_kw:', old_kw)
                syn_dict[syn] = composed_kw
        

def insert_dict(eid, kw_tup, dict, create_flag = False, nr_refs = 0): 
    if not (kw_tup in blackset): 
        if kw_tup in syn_dict:
            kw_tup = syn_dict[kw_tup]
        if kw_tup in dict:
            if not eid in dict[kw_tup].get_set():
                dict[kw_tup].add_to_set(eid, nr_refs)
        elif create_flag:
            dict[kw_tup] = Dictentry() # create a new entry 
            dict[kw_tup].add_to_set(eid)    

def clean_text(txt):
    sentence_splitters = ".!?"
    remove_characters = "',:;()[]" + '"' + "{" + "}"
    txt = txt.lower()
    for character in remove_characters + sentence_splitters:
        txt = txt.replace(character,' ')
    return txt


def increment_akw(eid, txt, dict, create_flag = False, nr_refs = 0):
    txt = clean_text(txt)
    sentence_list = txt.split('|') 
    for kw in sentence_list:
        insert_dict(eid,tuple(kw.split()) ,dict, create_flag, nr_refs = nr_refs)
    return len(sentence_list)

def increment_title_and_abstract(eid, txt, dict, is_title = True, nr_refs = 0):
    txt = txt = clean_text(txt)
    sentence_list = txt.split() 
    for kw_length in range(max_key_word_length):
        for position in range(len(sentence_list) - kw_length):
            if is_title or (kw_length > 0):
                insert_dict(eid,tuple(sentence_list[position:(position+kw_length+1)]), dict, nr_refs = nr_refs)
            
my_dictionary = {}

keyw_count = 0
no_keyw = 0
no_of_articles = 0

# pybliometrics.scopus.utils.create_config(["af6e11a84869dc42fe24d29d18c95c53"])
pybliometrics.scopus.utils.create_config(["09dce7e8cd19da50f853080a734d5151"])
# pybliometrics.scopus.utils.create_config(["98c848cb832c223ac14920639ccd6869"])
# pybliometrics.scopus.utils.create_config(["4e3b380a597abb76c84382b62c516222"])
# pybliometrics.scopus.utils.create_config(["1e7b37c2233d89a33e02c5779d21b294"])




# First pass: finding all key words

year_list_citations = []

for year in years: 
    search_str = first_part_of_search_str + str(year) + ')'
    print(search_str)
    s = pybliometrics.scopus.ScopusSearch(search_str, refresh = False)
    print("number of articles:", len(s.results))
    no_of_articles += len(s.results)
    no_of_citations = 0 
    documents_with_aff_country_this_year = 0
    for doc in s.results:
        if doc.authkeywords:
            keyw_count += 1
            # print("number of citations: ", doc.citedby_count)
            no_keyw += increment_akw(doc.eid, doc.authkeywords, my_dictionary, create_flag = True)
        if doc.affiliation_country:
                no_of_citations += doc.citedby_count
                documents_with_aff_country_this_year += 1
    # year_list_citations += [(year,no_of_citations,len(s.results),no_of_citations/len(s.results))]
    year_list_citations += [(year,no_of_citations,documents_with_aff_country_this_year,no_of_citations/documents_with_aff_country_this_year)]

print("Year list citation")
print(year_list_citations)

print("Number of documents:", no_of_articles)
print("Documents with keywords: ", keyw_count)
print("Number of keywords: ", no_keyw)
print("Number of unique keywords: ", len(my_dictionary))
print("Maximum key word length: ", max_key_word_length)

# Second pass: counting number of documents that mention each key word

sort_list = list(my_dictionary.items())
sort_list.sort(key = lambda x: x[1].get_count(), reverse = True)
i = 0
active_kws = 0
my_new_dictionary = {}
print("len(my_dictionary)", len(my_dictionary))
while sort_list[i][1].get_count() > minimum_kw_occaision: # only use key words that are present in 20 or more documents
    if (len(sort_list[i][0]) > 1) or (sort_list[i][1].get_count() > 4*minimum_kw_occaision):
        my_new_dictionary[sort_list[i][0]] = Dictentry()
        active_kws += 1
    i += 1 
my_dictionary = my_new_dictionary


print("Number of key words that are present in 30 or more documents: ", active_kws)

for j in range(min(i,800)):
    for string in sort_list[j][0]:
        print(string, end = ' ')
    print(sort_list[j][1].get_count())

input()


year_list = []
year_list_countries = []

# top_list = [('Data', 'processing'), ('Learning', 'paradigms'), ('ALGORITHMS',), ('System', 'and', 'hardware'), ('APPLICATIONS',)]
top_list = [('Data', 'mining'), ('Healthcare',), ('Social', 'media'), ('Security',), ('Image', 'processing', 'and', 'computer', 'vision'),('Natural', 'language', 'processing'),('Internet', 'of', 'things'), ('Other', 'applications')]
# top_list = [('NN',), ('Not', 'NN')]
# top_list = [('Data', 'processing'), ('Learning', 'paradigms'), ('NN',), ('Not', 'NN'), ('System', 'and', 'hardware'), ('APPLICATIONS',), ('reinforcement', 'learning')]
# top_list = [('Data', 'processing'), ('Learning', 'paradigms'), ('ALGORITHMS',), ('System', 'and', 'hardware'), ('APPLICATIONS',)]
# top_list = [('Data', 'processing'), ('Learning', 'paradigms'), ('System', 'and', 'hardware'), ('Data', 'mining'), ('Neural', 'network', 'algorithms'), ('Other', 'algorithms', 'than', 'neural', 'networks'), ('Healthcare',), ('Social', 'media'), ('Security',), ('Image', 'processing', 'and', 'computer', 'vision'),('Natural', 'language', 'processing'),('Internet', 'of', 'things')]
# top_list = [('CNN', ), ('RNN', ), ('Other', 'neural', 'network', 'algorithms')]
# top_list = [('reinforcement', 'learning')]
# top_list = [('Ensemble', ), ('Linear', ), ('Nearest', 'neighbor'), ('Kernel', 'based'), ('Tree', 'based'), ('Clustering',), ('Statistical',), ('Other', 'algorithms')]
# top_list = [('Neural', 'network', 'algorithms'), ('Other', 'algorithms', 'than', 'neural', 'networks')]


china_count = 0
china_accumulated = 0
china_total_citations = 0
china_kw_citation_list = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
us_count = 0
us_accumulated = 0
us_total_citations = 0
us_kw_citation_list = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
eu_count = 0
eu_accumulated = 0
eu_total_citations = 0
eu_kw_citation_list = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
other_count = 0
other_accumulated = 0
other_total_citations = 0
other_kw_citation_list = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

all_accumulated = 0
all_count = 0

number_of_docs_witout_aff_country = 0


european_countries = {"switzerland", "norway", "serbia", "ukraine", "bosnia-and-herzegovina", "albania", "iceland", "austria", "belgium", "bulgaria", "croatia", "republic-of-cyprus", "czech-republic", "denmark", "estonia", "finland", "france", "germany", "greece", "united-kingdom", "hungary", "ireland", "italy", "latvia", "lithuania", "luxembourg", "malta", "netherlands", "poland", "portugal", "romania", "slovakia", "slovenia", "spain", "sweden"}
brics_countries = {"china", "india", "brazil", "russian-federation", "south-africa"}
north_america_countries = {"united-states", "canada"}

old_year_list_citations = copy.deepcopy(year_list_citations)

for year in years:
    search_str = first_part_of_search_str + str(year) + ')'
    print(search_str)
    year_dictionary = copy.deepcopy(my_dictionary)
    citation_tup = year_list_citations.pop(0)

    print("Citation tuple:",citation_tup)

    s = pybliometrics.scopus.ScopusSearch(search_str, refresh = False)
    print("number of articles:", len(s.results))

    for doc in s.results:
        # Check title
        if doc.title:
            increment_title_and_abstract(doc.eid, doc.title, year_dictionary, nr_refs = doc.citedby_count)

        # Check key words
        if doc.authkeywords:
            increment_akw(doc.eid, doc.authkeywords, year_dictionary, nr_refs = doc.citedby_count)


        # Check abstract
        if doc.description:
            increment_title_and_abstract(doc.eid, doc.description, year_dictionary, is_title = False, nr_refs = doc.citedby_count)

        if doc.affiliation_country:
            all_accumulated += doc.citedby_count/citation_tup[3]
            all_count += 1
            aff_country = doc.affiliation_country
            aff_country = aff_country.replace(' ','-')
            countries = clean_text(aff_country)
            country_list = countries.split() 
            number_of_countries = len(country_list)
            for country in country_list:
                if country != 'none':
                    if country in brics_countries:
                        china_total_citations += doc.citedby_count/number_of_countries
                        china_accumulated += doc.citedby_count/citation_tup[3]/number_of_countries
                        china_count += 1/number_of_countries
                        for key_word in top_list:
                            if doc.eid in year_dictionary[key_word].get_set():
                                china_kw_citation_list[top_list.index(key_word)][0] += doc.citedby_count/number_of_countries
                                china_kw_citation_list[top_list.index(key_word)][1] += doc.citedby_count/citation_tup[3]/number_of_countries
                                china_kw_citation_list[top_list.index(key_word)][2] += 1/number_of_countries
                    elif  country in north_america_countries:
                        us_total_citations += doc.citedby_count/number_of_countries
                        us_accumulated += doc.citedby_count/citation_tup[3]/number_of_countries
                        us_count += 1/number_of_countries
                        for key_word in top_list:
                            if doc.eid in year_dictionary[key_word].get_set():
                                us_kw_citation_list[top_list.index(key_word)][0] += doc.citedby_count/number_of_countries
                                us_kw_citation_list[top_list.index(key_word)][1] += doc.citedby_count/citation_tup[3]/number_of_countries
                                us_kw_citation_list[top_list.index(key_word)][2] += 1/number_of_countries
                    elif country in european_countries:
                        eu_total_citations += doc.citedby_count/number_of_countries
                        eu_accumulated += doc.citedby_count/citation_tup[3]/number_of_countries
                        eu_count += 1/number_of_countries
                        for key_word in top_list:
                            if doc.eid in year_dictionary[key_word].get_set():
                                eu_kw_citation_list[top_list.index(key_word)][0] += doc.citedby_count/number_of_countries
                                eu_kw_citation_list[top_list.index(key_word)][1] += doc.citedby_count/citation_tup[3]/number_of_countries
                                eu_kw_citation_list[top_list.index(key_word)][2] += 1/number_of_countries
                    else:
                        # print(country)
                        other_total_citations += doc.citedby_count/number_of_countries
                        other_accumulated += doc.citedby_count/citation_tup[3]/number_of_countries
                        other_count += 1/number_of_countries
                        for key_word in top_list:
                            if doc.eid in year_dictionary[key_word].get_set():
                                other_kw_citation_list[top_list.index(key_word)][0] += doc.citedby_count/number_of_countries
                                other_kw_citation_list[top_list.index(key_word)][1] += doc.citedby_count/citation_tup[3]/number_of_countries
                                other_kw_citation_list[top_list.index(key_word)][2] += 1/number_of_countries
                else:
                    print("COUNTRY IS NONE", country)
        else:
            number_of_docs_witout_aff_country += 1
    year_list += [(year,year_dictionary)]
    year_list_countries += [(year,china_count,us_count,eu_count,other_count)]

print("year_list_countries", year_list_countries)
year_list_citations = old_year_list_citations

print()
print('number_of_docs_witout_aff_country',number_of_docs_witout_aff_country)
print()
print("Average China citations: ",round(china_accumulated/china_count,2),' ',round(china_count),' ',round(china_total_citations))
for key_word in top_list:
    print(key_word, end = ' ')
    print(round(china_kw_citation_list[top_list.index(key_word)][1]/china_kw_citation_list[top_list.index(key_word)][2],2),' ',round(china_kw_citation_list[top_list.index(key_word)][2]),' ',round(china_kw_citation_list[top_list.index(key_word)][0]) )
print("Average US citations: ",round(us_accumulated/us_count,2),' ',round(us_count),' ',round(us_total_citations))
for key_word in top_list:
    print(key_word, end = ' ')
    print(round(us_kw_citation_list[top_list.index(key_word)][1]/us_kw_citation_list[top_list.index(key_word)][2],2),' ',round(us_kw_citation_list[top_list.index(key_word)][2]),' ',round(us_kw_citation_list[top_list.index(key_word)][0]) )
print("Average EU citations: ",round(eu_accumulated/eu_count,2),' ',round(eu_count),' ', round(eu_total_citations))
for key_word in top_list:
    print(key_word, end = ' ')
    print(round(eu_kw_citation_list[top_list.index(key_word)][1]/eu_kw_citation_list[top_list.index(key_word)][2],2),' ',round(eu_kw_citation_list[top_list.index(key_word)][2]),' ',round(eu_kw_citation_list[top_list.index(key_word)][0]) )
print("Average Other citations: ",round(other_accumulated/other_count,2),' ',round(other_count),' ', round(other_total_citations))
for key_word in top_list:
    print(key_word, end = ' ')
    print(round(other_kw_citation_list[top_list.index(key_word)][1]/other_kw_citation_list[top_list.index(key_word)][2],2),' ',round(other_kw_citation_list[top_list.index(key_word)][2]),' ',round(other_kw_citation_list[top_list.index(key_word)][0]) )
print("Average for top keywords")
for key_word in top_list:
    print(key_word, end = ' ')
    print(round((china_kw_citation_list[top_list.index(key_word)][1] + us_kw_citation_list[top_list.index(key_word)][1] + eu_kw_citation_list[top_list.index(key_word)][1] + other_kw_citation_list[top_list.index(key_word)][1])/(china_kw_citation_list[top_list.index(key_word)][2] + us_kw_citation_list[top_list.index(key_word)][2] + eu_kw_citation_list[top_list.index(key_word)][2] + other_kw_citation_list[top_list.index(key_word)][2]),2))
print("Average All citations: ",all_accumulated/all_count,' ',all_count)
print()

# accumulate results

current_age_factor = 1
total_dictionary = copy.deepcopy(my_dictionary)
age_dictionary = copy.deepcopy(my_dictionary)
citation_dictionary = copy.deepcopy(my_dictionary)
total_citation_dictionary = copy.deepcopy(my_dictionary)

ranking_dictionary = copy.deepcopy(my_dictionary)

for year_tup in year_list:
    for dict_key in (year_tup[1]):
        total_dictionary[dict_key].add_age_counter(year_tup[1][dict_key].get_count())
        total_dictionary[dict_key].make_union(year_tup[1][dict_key].get_set())
        
previous_year_tup = ()
for year_tup in year_list:
    if not year_tup[0] == years[0]: # not first year
        for dict_key in (year_tup[1]):
            age_dictionary[dict_key].add_age_counter((year_tup[1][dict_key].get_count() - previous_year_tup[1][dict_key].get_count())*current_age_factor)
    previous_year_tup = year_tup
    current_age_factor *= age_factor

for year_tup in year_list:
    for dict_key in (year_tup[1]):
        total_citation_dictionary[dict_key].add_age_counter(year_tup[1][dict_key].get_citations())

old_year_list_citations = copy.deepcopy(year_list_citations)

for year_tup in year_list:
    citation_tup = year_list_citations.pop(0)
    print("Citation tuple: ", citation_tup)
    for dict_key in (year_tup[1]):
        if total_dictionary[dict_key].get_age_counter() == 0:
            print("Error: ", dict_key)
        if year_tup[1][dict_key].get_count() > 0:
            citation_dictionary[dict_key].add_age_counter(((year_tup[1][dict_key].get_citations()/year_tup[1][dict_key].get_count())/citation_tup[3])*(year_tup[1][dict_key].get_count()/total_dictionary[dict_key].get_age_counter()))

year_list_citations = old_year_list_citations

# Print results

print()
print("Top list")
print(top_list)

    
import csv
f = open('outfile.csv', 'w')
writer = csv.writer(f)

print("Number of articles per keyword and year")

for first_kw in top_list:
    print(first_kw)
    row = [first_kw]
    for year_tup in year_list:
        print(year_tup[1][first_kw].get_count(), end =', ')
        row += [year_tup[1][first_kw].get_count()]
    print()
    writer.writerow(row)
f.close()


