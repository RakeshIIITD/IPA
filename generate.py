from __future__ import division
from Module3 import *
from Module1 import *
import math
import random
import sys
import os
import itertools
import operator
import random
import string
import nltk
from nltk.corpus import brown as bw                       # corpus for different genres
from nltk.corpus import wordnet as wn                     # corpus for structured words
from nltk.corpus import stopwords                         # corpus for stopwords
from nltk.corpus import gutenberg as gut                  # corpus for e-books
from nltk.corpus import nps_chat as chat                  # corpus for chats
from nltk.corpus import inaugural as president_speeches   # corpus for news/speeches

#function to find mod in a list
def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

# function finds all possible lookhead letter combinations according to given order 
def manp_possbls( order ):
    lst=[]
    for k in range(0,pow(26,order)):
        count = 1
        comb = ''
        for letter in range(0,order): 
            comb = comb + get_char(((int(k/(pow(26,order-count))))%26)+1)
            count+=1
        lst.append([comb])
    return lst       

#function finds index respective character      
def get_char(index):
    letter_string = string.ascii_lowercase
    return letter_string[index-1]   


#os.system('CLS')

ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85
#def mod_2(SEEDS, DOB, ORDER, FAV_MOVIE_GENRE, NEWS_FLAG, FAV_HOBBY, OCCUPATION, CHAT_FLAG, BOOK_FLAG)
#mod_2(SEEDS, DOB, ORDER, FAV_MOVIE_GENRE, NEWS_FLAG, FAV_HOBBY, OCCUPATION, CHAT_FLAG, BOOK_FLAG):
ORDER = 2
print "Enter the following choices: "
print "Your name: "
nname = raw_input();
print "Enter a keyword of your choice: "
kkey = raw_input();
print "Enter your DOB: "
DOB = raw_input();
print "Enter your favourite move genre: \n\t1.adventure\n\t2.humor\n\t3.mystery\n\t4.romance\n\t5.science_fiction\n"
FAV_MOVIE_GENRE = int(raw_input())
print "Do you like news??(y/n): "
NEWS_FLAG = raw_input()
print "Enter your favourite hobby: "
FAV_HOBBY = raw_input()
print "Enter your occupation: "
OCCUPATION = raw_input()
print "Do you like chatting??(y/n): "
CHAT_FLAG = raw_input()
print "Do you like reading books??(y/n): "
BOOK_FLAG = raw_input()
SEEDS = sha512(nname,kkey)
#test_seed = int(SEEDS)

def get_best_synset_pair(word_1, word_2):
    """ 
    Choose the pair with highest path similarity among all pairs. 
    Mimics pattern-seeking behavior of humans.
    """
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
               sim = wn.path_similarity(synset_1, synset_2)
               if sim > max_sim:
                   max_sim = sim
                   best_pair = synset_1, synset_2
        return best_pair


def length_dist(synset_1, synset_2):
    """
    Return a measure of the length of the shortest path in the semantic 
    ontology (Wordnet in our case as well as the paper's) between two 
    synsets.
    """
    l_dist = sys.maxint
    if synset_1 is None or synset_2 is None: 

        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])        
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)


def hierarchy_dist(synset_1, synset_2):
    """
    Return a measure of depth in the ontology to model the fact that 
    nodes closer to the root are broader and have less semantic similarity
    than nodes further away from the root.
    """
    h_dist = sys.maxint
    if synset_1 is None or synset_2 is None: 
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if hypernyms_1.has_key(lcs_candidate):
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if hypernyms_2.has_key(lcs_candidate):

                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / 
        (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))


def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) * 
        hierarchy_dist(synset_pair[0], synset_pair[1]))


#def mod_2(SEEDS, DOB, ORDER, FAV_MOVIE_GENRE, NEWS_FLAG, FAV_HOBBY, OCCUPATION, CHAT_FLAG, BOOK_FLAG):
###############################################################################
##################### -----WORD SIMILARITY FUNCTIONS----- #####################


# Parameters to the algorithm. Currently set to values that was reported
# in the paper to produce "best" results.
	

###############################################################################
###############################################################################

brown_freqs = dict()
N = 0


test_seed = int(SEEDS)
print "seeds extracted..."


movie_genres = {

    
        	'1':'adventure',
        	'2':'humor',
        	'3':'mystery',
        	'4':'romance',
        	'5':'science_fiction'


                }


#shakespeare_books = {
#
#        '1':'14',
#        '2':'15',
#        '3':'16',
#                    }


#FAV_MOVIE_GENRE = int(raw_input('Q1. What is your favourite movie type?(1, 2, 3, 4 or 5): \
#                               \n\t1.Adventure\n\t2.Humor\n\t3.Mystery\
#                              \n\t4.Romatic\n\t5.Science Fiction\n')) #----brown(genres)------

#NEWS_FLAG = raw_input('Q2. Do you like news?(Y/N) : ')     
NEWS_FLAG = NEWS_FLAG.lower()                                           #----- inaugral---------

#FAV_HOBBY = raw_input('Q3. Which is your favourite hobby? : ')          #-----brown(hobby)------
FAV_HOBBY = FAV_HOBBY.lower()

#OCCUPATION = raw_input('Q4. What is your occupation? : ')               #-----brown(all words)--
OCCUPATION = OCCUPATION.lower()

#CHAT_FLAG = raw_input('Q5. Do you like chatting?(Y/N) : ')              #------nps_chat---------         
CHAT_FLAG = CHAT_FLAG.lower()

#BOOK_FLAG = raw_input('Q6. Do you like Shakespeares\' books?(Y/N) : ')  #-----gutenberg---------
#BOOK_FLAG = raw_input('Q6. Do you like reading books?(Y/N) : ')  #-----gutenberg---------
BOOK_FLAG = BOOK_FLAG.lower()

#if BOOK_FLAG == 'y':
#    FAV_BOOK = int(raw_input('Which book will you prefer? :(1, 2 or 3) \
#    \n\t1.Julius Caesar\n\t2.Hamlet\n\t3.Macbeth\n'))


related_words_MOVIE = []
related_words_NEWS = []
related_words_HOBBY = []
related_words_OCCUPATION = []
related_words_CHAT = []
related_words_BOOK = []


################## Filtering of stopwords from corpuses ##################

stop = set(stopwords.words('english'))

##--------------------------- MOVIE ---------------------------
#
#movie_cat = movie_genres.get(str(FAV_MOVIE_GENRE))
#movie_words = list(bw.words(categories = movie_cat))
#
#R1=random.getstate()[1][(long(test_seed%10000))%625]
#random.shuffle(movie_words, lambda: 1/R1)            # deterministic shuffling using seeds
#test_seed/=10000        
#
#filtered_movie_words = list(set(movie_words)-stop)
#[related_words_MOVIE.append(i) for i in filtered_movie_words if len(i)>4]
#related_words_MOVIE = related_words_MOVIE[0:2000]
#
##------------------------------------------------------------


#--------------------------- NEWS ---------------------------
print "starting extracted news data...\n"
if NEWS_FLAG != 'n':
#        print "processing news!!!"
   		news_words = list(president_speeches.words())
   		R0=random.getstate()[1][(long(test_seed%10000))%625]
   		random.shuffle(news_words, lambda: 1/R0)        # deterministic shuffling using seeds
   		test_seed/=10000   
   		filtered_news_words = list(set(news_words)-stop)
   		[related_words_NEWS.append(i) for i in filtered_news_words if len(i)>3]
   		related_words_NEWS = related_words_NEWS[0:2000]
print "finished extracted news data...\n"

#------------------------------------------------------------
   
   
#--------------------------- CHAT ---------------------------
print "starting extracted chat data...\n"

if CHAT_FLAG != 'n':
   		chat_words = list(chat.words())
  
   		R1=random.getstate()[1][(long(test_seed%10000))%625]
   		random.shuffle(chat_words, lambda: 1/R1)       # deterministic shuffling using seeds
   		test_seed/=10000
   
   		filtered_chat_words = list(set(chat_words)-stop)
   		[related_words_CHAT.append(i) for i in filtered_chat_words]
   		related_words_CHAT = related_words_CHAT[0:2000]
print "finished extracted chat data...\n"
    

#------------------------------------------------------------

#if BOOK_FLAG != 'n':
#    book_words = gut.words(gut.fileids()[int(shakespeare_books.get(str(FAV_BOOK)))])
#    filtered_book_words = set(book_words)-stop
#    filtered_book_words = list(filtered_book_words)


#---------------------------- BOOK --------------------------
print "starting extracted book data...\n"

if BOOK_FLAG != 'n':
   		book_words = list(gut.words())
   
   		R2=random.getstate()[1][(long(test_seed%10000))%625]
   		random.shuffle(book_words, lambda: 1/R2)       # deterministic shuffling using seeds
   		test_seed/=10000
   
   		filtered_book_words = list(set(book_words)-stop)
   		[related_words_BOOK.append(i) for i in filtered_book_words if len(i)>3]
   		related_words_BOOK = related_words_BOOK[0:2000]

print "finished extracted book data...\n"

#------------------------------------------------------------

############################ Filtering done #################################
#############################################################################



######################### Filtering by Similarity ###########################


#------------------------------ Occupation ---------------------------------
print "starting extracted occupation data...\n"

general_words_OCC = list(bw.words())

related_words_with_similarity_OCC=[]


R3=random.getstate()[1][(long(test_seed%10000))%625]
random.shuffle(general_words_OCC, lambda: 1/R3)         # deterministic shuffling using seeds
test_seed/=10000

filtered_general_words_OCC = list(set(general_words_OCC)-stop)
print "processing occupation..."
for word_OCC in filtered_general_words_OCC[:2000]:
	word_OCC.lower()    
  	if wn.synsets(word_OCC) == [] or len(word_OCC)<5:
  		continue
   	else:
   		related_words_with_similarity_OCC.append([word_similarity(OCCUPATION,word_OCC),word_OCC])
     

	related_words_with_similarity_OCC.sort()     
	related_words_with_similarity_OCC.reverse()
	for i in related_words_with_similarity_OCC:
    		related_words_OCCUPATION.append(i[1])

print "finished extracted occupation data...\n"

#--------------------------------------------------------------------------




#------------------------------ Hobby ---------------------------------
print "starting extracted hobby data...\n"

general_words_HOBBY = list(bw.words(categories='hobbies'))

related_words_with_similarity_HOBBY=[]


R4=random.getstate()[1][(long(test_seed%10000))%625]
random.shuffle(general_words_HOBBY, lambda: 1/R4)         # deterministic shuffling using seeds
test_seed/=10000

filtered_general_words_HOBBY = list(set(general_words_HOBBY)-stop)

for word_HOBBY in filtered_general_words_HOBBY[:2000]:
 	word_HOBBY.lower()    
   	if wn.synsets(word_HOBBY) == [] or len(word_HOBBY)<4:
   		continue
   	else:
   		related_words_with_similarity_HOBBY.append([word_similarity(FAV_HOBBY,word_HOBBY),word_HOBBY])
     

related_words_with_similarity_HOBBY.sort()     
related_words_with_similarity_HOBBY.reverse()
for i in related_words_with_similarity_HOBBY:
   		related_words_HOBBY.append(i[1])
print "finished extracted hobby data...\n"

#--------------------------------------------------------------------------


#------------------------------ Movie Genres ------------------------------
print "starting extracted movie genre data...\n"

mov_cat = movie_genres.get(str(FAV_MOVIE_GENRE))
general_words_MOVIE = list(bw.words(categories = mov_cat))

related_words_with_similarity_MOVIE=[]

if mov_cat == 'science_fiction':
   		mov_cat = 'gadgets'
    
R5=random.getstate()[1][(long(test_seed%10000))%625]
random.shuffle(general_words_MOVIE, lambda: 1/R5)         # deterministic shuffling using seeds
test_seed/=10000

filtered_general_words_MOVIE = list(set(general_words_MOVIE)-stop)

for word_MOVIE in filtered_general_words_MOVIE[:2000]:
   	word_MOVIE.lower()    
   	if wn.synsets(word_MOVIE) == [] or len(word_MOVIE)<5:
   		continue
   	else:	
		print word_MOVIE
   		related_words_with_similarity_MOVIE.append([word_similarity(mov_cat,word_MOVIE),word_MOVIE])
     

related_words_with_similarity_MOVIE.sort()     
related_words_with_similarity_MOVIE.reverse()
for i in related_words_with_similarity_MOVIE:
   		related_words_MOVIE.append(i[1])

print "finished extracted movie genre data...\n"

#--------------------------------------------------------------------------




#------------------------- MIXED NEW CORPORA -----------------------------

print "making mixed corpora\n"

MIXED = []
MIXED.extend(related_words_BOOK)
MIXED.extend(related_words_CHAT)
MIXED.extend(related_words_HOBBY)
MIXED.extend(related_words_MOVIE)
MIXED.extend(related_words_NEWS)
MIXED.extend(related_words_OCCUPATION)

R6=random.getstate()[1][(long(test_seed%10000))%625]
random.shuffle(MIXED, lambda: 1/R6)         # deterministic shuffling using seeds
test_seed/=10000

R7=random.getstate()[1][(long(test_seed%10000))%625]
random.shuffle(MIXED, lambda: 1/R7)         # deterministic shuffling using seeds (Reshuffling)
test_seed/=10000


print "mixed corpora successfully...\n"

#-------------------------------------------------------------------------

######markov implementation##########
	
print "applying markov assumption...\n"

similar_data = MIXED
order=int(ORDER)
hash_now=str(SEEDS)
# determining the random factor for seeding in random generator 
sum=0
for i in list(hash_now):
	sum = sum + int(i)
	
# converting to floating point for more accurate precision 

sum=float(sum)
sum = sum/((len(hash_now))*10)

if order==0:
                                                  
	random.shuffle(similar_data, lambda: sum)    # passing seeds with actual similar words list to random organization of words
	index = float(len(similar_data))*sum 
	word = similar_data[int(index)-1]
	tmp = sum
	word = word.encode('ascii','ignore')	


	if len(word) <= 4:                                     #Filteration on retrieved password for length constraints
		bias = (1-sum)/2 
		while bias>.01:
			tmp = tmp + bias
			index = float(len(similar_data))*(tmp) 
			word = similar_data[int(index)-1]
			if len(''.join(word)) > 4:
				break;
			bias = bias/2

ccc=0
ccc1=0
ccc2=0
ccc3=0
if order>=1:
	possbl_comb = manp_possbls(order)
	for comb in range(0,pow(26,order)):
#        print ccc
#        print ".\n"
#        ccc=ccc+1
		for occur in similar_data:
#            print ccc1
#            ccc1=ccc1+1
#            print "\n"
			occur = occur.encode('ascii','ignore')
			ind_match = occur.find(possbl_comb[comb][0]) # matching order combination string 
			if ind_match != -1 and (ind_match+order) < len(occur):
				possbl_comb[comb].append(occur[ind_match+order]) # storing matches as they matches in words

		print "In finding phras"
#        print ccc
#        ccc=ccc+1
		tmp_list = []	
		for i in range(0,26):
#            print ccc2
#            ccc2=ccc2+1
#            print "\n"
			tmp_char=get_char(i+1)
			tmp_list.append([possbl_comb[comb][1:].count(tmp_char),tmp_char]) # manipulating a temporary count for extracting only top most frequencies
		tmp_list.sort()	
		for k in range(0,(26-int(sum*26))):
#            print ccc3
#            ccc3=ccc3+1
#            print "\n"
			if tmp_list[k][0] != 0:
				while tmp_list[k][1] in possbl_comb[comb]:possbl_comb[comb].remove(tmp_list[k][1])

				
		
	len_psswd=int(sum*10+2)

	word = similar_data[(int(sum*len(similar_data)))-1].encode('ascii','ignore')
	word = word[:order]
	for k in range(0,len_psswd-order):
		tmp_range = int(len(hash_now)/(len_psswd-1))
		tmp_hash = hash_now[k*tmp_range:(k+1)*tmp_range]
		hash_mod = tmp_hash.count(most_common(tmp_hash))
		fraction_mod = float(float(hash_mod)/tmp_range)
		for i in possbl_comb:
			if word[-order:] == i[0] and len(i[1:]) != 0:
				word=word+i[1:][int(fraction_mod*(len(i)-1))]
				break
           


#Applying modification to maintain length complexities in password
print "markov finished...\n"

if len(word) <= 7:
	if sum>0.45:
		word = ''.join(list(word)+ list(DOB[(int(sum*len(DOB))-1)%len(DOB)]) + list(DOB[(int(sum*len(DOB)))%len(DOB)]) + list(DOB[(int(sum*len(DOB))+1)%len(DOB)]) + list(DOB[(int(sum*len(DOB))+2)%len(DOB)]))
	if sum<=0.45:
		word =''.join(list(DOB[(int(sum*len(DOB))-1)%len(DOB)]) + list(DOB[(int(sum*len(DOB)))%len(DOB)]) + list(DOB[(int(sum*len(DOB))+1)%len(DOB)]) + list(DOB[(int(sum*len(DOB))+2)%len(DOB)]) + list(word))	

print "=====>GENERATED PASSWORD<=====\n"
print module_substitute(word,int(test_seed))
print "\n================================\n"
'''
#function to find mod in a list
def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

# function finds all possible lookhead letter combinations according to given order 
def manp_possbls( order ):
	lst=[]
	for k in range(0,pow(26,order)):
		count = 1
		comb = ''
		for letter in range(0,order): 
			comb = comb + get_char(((int(k/(pow(26,order-count))))%26)+1)
			count+=1
		lst.append([comb])
	return lst		 

#function finds index respective character		
def get_char(index):
	letter_string = string.ascii_lowercase
	return letter_string[index-1]	

'''


'''
#mod_2(SEEDS, DOB, ORDER, FAV_MOVIE_GENRE, NEWS_FLAG, FAV_HOBBY, OCCUPATION, CHAT_FLAG, BOOK_FLAG):
print "Enter the following choices: "
print "Your name: "
nname = raw_input();
print "Enter a keyword of your choice: "
kkey = raw_input();
print "Enter your DOB: "
ddob = raw_input();
print "Enter your favourite move genre: \n\t1.adventure\n\t2.humor\n\t3.mystery\n\t4.romance\n\t5.science_fiction\n"
ggenre = int(raw_input())
print "Do you like news??(y/n): "
nnews = raw_input()
print "Enter your favourite hobby: "
hhobby = raw_input()
print "Enter your occupation: "
ooccupation = raw_input()
print "Do you like chatting??(y/n): "
cchat = raw_input()
print "Do you like reading books??(y/n): "
bboks = raw_input()
hash_value = sha512(nname,kkey)
print mod_2(int(hash_value),ddob,2,ggenre,nnews,hhobby,ooccupation,cchat,bboks)'''
#print mod_2(int(hash_value),ddob,2,ggenre,nnews,hhobby,ooccupation,cchat,bboks)
