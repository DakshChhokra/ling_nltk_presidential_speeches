from textblob import TextBlob
import nltk
from newspaper import Article
# nltk.download('vader_lexicon')
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
sid = SentimentIntensityAnalyzer()


def get_sentence_array(url):
	article = Article(url)
	article.download()
	article.parse()
	article.nlp()
	lines_list = tokenize.sent_tokenize(article.text)
	return lines_list

	

# sentences = []
# kennedy_url = 'https://americanrhetoric.com/speeches/jfkriceuniversity.htm'
# reagan_url = 'http://www.atomicarchive.com/Docs/Missile/Starwars.shtml'

# k_arr = get_sentence_array(kennedy_url)
# r_arr = get_sentence_array(reagan_url)

# sid = SentimentIntensityAnalyzer()
# k_comp = 0
# k_comp_count = 0;
# for sentence in k_arr:
# 	ss = sid.polarity_scores(sentence)
# 	k_comp = k_comp + ss.get('compound')
# 	k_comp_count += 1
# average_k_comp = k_comp/k_comp_count
# print("kennedy's sentiemnt is {}".format(average_k_comp))

# r_comp = 0
# r_comp_count = 0;
# for sentence in r_arr:
# 	ss = sid.polarity_scores(sentence)
# 	r_comp = r_comp + ss.get('compound')
# 	r_comp_count += 1
# average_r_comp = r_comp/r_comp_count
# print("reagan's sentiement is {}".format(average_r_comp))

print("###############################")
def get_compound_of_speech(sentences):
	comp = 0
	comp_count = 0
	for sentence in sentences:
		ss = sid.polarity_scores(sentence)
		comp += ss.get('compound')
		comp_count += 1
	return comp/comp_count

kennedy_first_ten_urls = ["https://www.americanrhetoric.com/speeches/jfk1960dnc.htm", "https://www.americanrhetoric.com/speeches/jfkhoustonministers.html", "https://www.americanrhetoric.com/speeches/jfkopeningstatementnixondebate1.htm", "https://www.americanrhetoric.com/speeches/jfkcommonwealthmass.htm", "https://www.americanrhetoric.com/speeches/jfkinaugural.htm", "https://www.americanrhetoric.com/speeches/jfkpeacecorpsexecutiveorder.htm", "https://www.americanrhetoric.com/speeches/jfklatinamericadiplomaticcore.htm", "https://www.americanrhetoric.com/speeches/jfksocietyofnewspapereditors.htm", "https://www.americanrhetoric.com/speeches/jfknewspaperpublishers.htm", "https://www.americanrhetoric.com/speeches/jfkjointsessionurgentnationalneeds.htm"]
kennedy_last_ten_urls = ["https://www.americanrhetoric.com/speeches/jfkundelivereddallasspeech.htm", "https://www.americanrhetoric.com/speeches/jfkfortworthcocommerce.htm", "https://www.americanrhetoric.com/speeches/jfksanantoniomedicalcenter.htm", "https://www.americanrhetoric.com/speeches/jfkmormontabernacle.htm", "https://www.americanrhetoric.com/speeches/jfknucleartestbantreaty.htm", "https://www.americanrhetoric.com/speeches/jfkunitednations1963.htm", "https://www.americanrhetoric.com/speeches/jfkairforceacademycommencement.htm", "https://www.americanrhetoric.com/speeches/jfkirishparliament.htm", "https://www.americanrhetoric.com/speeches/jfkberliner.html", "https://www.americanrhetoric.com/speeches/jfkcivilrights.htm"]
reagan_first_ten_urls  = ["https://www.americanrhetoric.com/speeches/ronaldreagansocializedmedicine.htm", "https://www.americanrhetoric.com/speeches/ronaldreaganatimeforchoosing.htm", "https://www.americanrhetoric.com/speeches/ronaldreagancalgovcandidacy.htm", "https://www.americanrhetoric.com/speeches/ronaldreagan1976rnc.htm", "https://www.americanrhetoric.com/speeches/ronaldreaganhillsdalecollege.htm", "https://www.americanrhetoric.com/speeches/ronaldreaganreligiousliberty.htm", "https://www.americanrhetoric.com/speeches/ronaldreaganlibertypark.htm", "https://www.americanrhetoric.com/speeches/ronaldreagan1980rnc.htm", "https://www.americanrhetoric.com/speeches/ronaldreaganpresidentelectvictory.htm", "https://www.americanrhetoric.com/speeches/ronaldreagandfirstinaugural.html"]
reagan_last_ten_urls = ["https://www.americanrhetoric.com/speeches/ronaldreagan83rdbirthday.htm", "https://www.americanrhetoric.com/speeches/ronaldreaganfarewelladdress.html", "https://www.americanrhetoric.com/speeches/ronaldreaganvietnammemorial.html", "https://www.americanrhetoric.com/speeches/ronaldreaganmoscowstateuniversity.htm", "https://www.americanrhetoric.com/speeches/ronaldreaganbrandenburggate.htm", "https://www.americanrhetoric.com/speeches/ronaldreaganirancontraspeech.htm", "https://www.americanrhetoric.com/speeches/ronaldreagantaxreformactof1986.html", "https://www.americanrhetoric.com/speeches/ronaldreagannsanewfacilitiesdedication.htm", "https://www.americanrhetoric.com/speeches/ronaldreaganchallenger.htm", "https://www.americanrhetoric.com/speeches/ronaldreaganbergen-belsen.htm"]

reagan_last_ten_urls.reverse()
reagan_urls = []
reagan_urls.extend(reagan_first_ten_urls);
reagan_urls.extend(reagan_last_ten_urls);

kennedy_urls = []
kennedy_urls.extend(kennedy_first_ten_urls);
kennedy_urls.extend(kennedy_last_ten_urls);


kennedy_compounds = []
reagan_compounds = []

for speech in kennedy_urls:
	curr_speech = get_sentence_array(speech)
	kennedy_compounds.append(get_compound_of_speech(curr_speech))

for speech in reagan_urls:
	curr_speech = get_sentence_array(speech)
	reagan_compounds.append(get_compound_of_speech(curr_speech))

print("Kennedy's sentiments were: ")
for k in kennedy_compounds:
	if k < 0:
		print(kennedy_urls[kennedy_compounds.index(k)])
	print(k)
print("Kennedy's average sentiment was ", sum(kennedy_compounds) / len(kennedy_compounds))
print("Kennedy's median sentiment was", np.median(kennedy_compounds))
print("Kennedy's std dev was", np.std(kennedy_compounds))
print("max: ", max(kennedy_compounds))
print("min: ", min(kennedy_compounds))

print("Reagan's sentiments were: ")
for r in reagan_compounds:
	if r < 0:
		print(reagan_urls[reagan_compounds.index(r)])
	print(r)
print("Reagan's average sentiment was ", sum(reagan_compounds) / len(reagan_compounds))
print("Reagan's median sentiment was", np.median(reagan_compounds))
print("Reagan's std dev was", np.std(reagan_compounds))
print("max: ", max(reagan_compounds))
print("min: ", min(reagan_compounds))

speech_numbers = []
for x in range(1, 21):
	speech_numbers.append(x)





plt.clf()

# using some dummy data for this example
xs = speech_numbers
ys = kennedy_compounds

plt.bar(xs,ys)

# zip joins x and y coordinates in pairs
for x,y in zip(xs,ys):

    label = "{:.4f}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xticks(np.arange(1,21,1))
plt.yticks(np.arange(-0.2,0.4,0.05))
plt.xlabel("Speech Number in Corpus")
plt.ylabel("VADER Sentiment Analysis Compound Score")

plt.show()

# using some dummy data for this example
xs = speech_numbers
ys = reagan_compounds

plt.bar(xs,ys)

# zip joins x and y coordinates in pairs
for x,y in zip(xs,ys):

    label = "{:.4f}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xticks(np.arange(1,21,1))
plt.yticks(np.arange(-0.2,0.4,0.05))
plt.xlabel("Speech Number in Corpus")
plt.ylabel("VADER Sentiment Analysis Compoun Score")

plt.show()



#ANALYSIS 2

k_len = []
r_len = []

def sentence_length_k(arr):
	for sentence in arr:
		k_len.append(len(sentence))

def sentence_length_r(arr):
	for sentence in arr:
		r_len.append(len(sentence))

for speech in kennedy_urls:
	curr_speech = sentence_length_k(get_sentence_array(speech))

for speech in reagan_urls:
	curr_speech = sentence_length_r(get_sentence_array(speech))


print("Kennedy's average sentence lengths were ", sum(k_len) / len(k_len))
print("Kennedy's median sentence length was", np.median(k_len))
print("Kennedy's std dev was", np.std(k_len))
print("max: ", max(k_len))
print("min: ", min(k_len))

print("Reagan's senntence lengths were: ")
print("Reagan's average sentence length was ", sum(r_len) / len(r_len))
print("Reagan's median sentence length was", np.median(r_len))
print("Reagan's std dev was", np.std(r_len))
print("max: ", max(r_len))
print("min: ", min(r_len))






