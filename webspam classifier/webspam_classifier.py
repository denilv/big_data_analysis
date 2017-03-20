from include import *
from sklearn.utils import shuffle
import pickle
from stop_words import get_stop_words
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
import zlib
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
# get_ipython().magic(u'matplotlib notebook')

TRAIN_DATA_FILE  = 'kaggle/kaggle_train_data_tab.csv.gz'
train_docs = file2docs_csv(TRAIN_DATA_FILE, reparse=False)

def calc_stats(words, url):
    title = words[0]
    text = words[1]
    lens = []
    n_refs = 0
    n_scripts = 0
    n_imgs = 0
    n_dots = url.count('.')
    n_slashes = url.count('/')
    n_nums = sum(c.isdigit() for c in url)
    domen = url.split("//")[-1].split("/")[0]
    domen_len = len(domen)
    domen_nums = sum(c.isdigit() for c in domen)
    n_hyphs = url.count('-')
    for i in text:
        if i in ('http', 'https', u'http', u'https'):
            n_refs += 1
        if i == 'script' or i == u'script':
            n_scripts += 1
        if i in ('png', 'jpg', 'raw', 'tiff', u'png', u'jpg', u'raw', u'tiff'):
            n_imgs += 1
        lens.append(len(i))
        # print n_imgs
    mean = float(np.sum(lens)) / len(text)

    big_sentence = " ".join(text)
    encoded = big_sentence.encode('utf-8')
#     print encoded
    encodeb64 = encoded.encode('zip')

    compress_coef = float(len(encodeb64)) / len(encoded)
    compress_coef2 = float(len(zlib.compress(encoded))) / len(encoded)

    most_common_word_len = np.bincount(lens).argmax()



    return [len(text), mean, len(title), most_common_word_len, compress_coef, compress_coef2, n_refs, n_scripts, n_imgs,\
            n_dots, n_slashes, n_nums, n_hyphs, domen_len, domen_nums]

def split_train(x, size = 5000):
    xs = shuffle(x)
    xtrain = xs[:size]
    xtest = xs[size:]
    return xtrain, xtest

train_docs, test_docs = split_train(train_docs, size = 7000)
print "Splitted"

alltext = []
for ind, doc in enumerate(train_docs):
    doc_id, is_spam, url, words = doc
    alltext.append(words[1])

train_text = []
for i in alltext:
    train_text.append(" ".join(i))

alltext = []
for ind, doc in enumerate(test_docs):
    doc_id, is_spam, url, words = doc
    alltext.append(words[1])

test_text = []
for i in alltext:
    test_text.append(" ".join(i))
print "Texted"

v = TfidfVectorizer(min_df = 50, max_features=10000, max_df = 5000)#, stop_words=get_stop_words('russian')) #max_df=len(doc_strs) * 0.7)
# v = TfidfVectorizer(stop_words=get_stop_words('russian'), max_features=1000)
vtrain = v.fit_transform(train_text).toarray()#TODO change it
print len(v.get_feature_names())
print "Vectorized"

xtrain = []
ytrain = []
for ind, doc in enumerate(train_docs):
    doc_id, is_spam, url, words = doc
    ytrain.append(is_spam)
    xtrain.append(calc_stats(words, url) + vtrain[ind].tolist())

xtest = []
ytest = []
vtest = v.transform(test_text).toarray()
for ind, doc in enumerate(test_docs):
    doc_id, is_spam, url, words = doc
    ytest.append(is_spam)
    xtest.append(calc_stats(words, url) + vtest[ind].tolist())
print "Text features"

del test_docs, train_docs, test_text, train_text

def check_cl(cl):
    cl = Classifier(cl)
    cl.train(xtrain, ytrain)

    preds = cl.predict_all(xtest)
    max_th = 0.0
    max_f1 = 0.0
    max_acc = 0.0
    for th in np.arange(0.3, 0.7, 0.005):
        ypred = [1 if pred > th else 0 for pred in preds]
        acc = accuracy_score(ytest, ypred)
        f1 = f1_score(ytest, ypred)
        # print "Th: ", th, "Acc: ", acc
        if (f1 > max_f1):
            max_th = th
            max_acc = acc
            max_f1 = f1
    print "max f1: ", max_f1, "max_acc: ", max_acc, "max_th: ", max_th
    return [cl, max_th, max_f1, max_acc]

cl_th_acc = []
# for l in ('l1', 'l2'):
#     for C in (0.01, 0.1, 1.0, 10.0, 20.0, 50.0, 100.0):
#         print "C: ", C
#         cl = LogisticRegression(n_jobs=-1, C=C, penalty=l)
#         cl_th_acc.append(check_cl(cl))

# cl = LogisticRegressionCV(verbose=4, n_jobs=-1, Cs=(0.01, 0.1, 1, 10, 20, 50, 100))
# cl.fit(xtrain, ytrain)
# print cl.score(xtest, ytest)

# cl = LogisticRegression(n_jobs=-1)
# params = {'C':(1, 10, 20, 50, 100, 200)}#{'C':(10, 20, 50, 100, 200, 500, 1000)}
# # params = {'C':(10,)}
# cl = GradientBoostingClassifier()
# params = {'learning_rate' : [0.1, 0,01], 'n_estimators' : [100, 500]}
# grid = GridSearchCV(cl, params, n_jobs=1, verbose=3)
# grid.fit(xtrain, ytrain)
# print '*'*80
# print "Results"
# print grid.best_estimator_
# cl = grid.best_estimator_
# cl = LogisticRegression(C=10, n_jobs=-1)
cl = GradientBoostingClassifier(n_estimators=700, verbose=3)
cl.fit(xtrain, ytrain)
print "Fitted"
f = open('gboostnew.pckl', 'wb')
pickle.dump(cl, f)
print "Dumped"

# npa = np.array(cl_th_acc)
# maxim = np.argmax(npa[:,2])
# cl, max_th, max_f1, max_acc = npa[maxim]
# print('*' * 80)
# print "Max th, max f1, max acc", max_th, max_f1, max_acc

del xtest, xtrain, ytest, ytrain, vtest, vtrain, alltext

if (len(sys.argv) > 1):
    with open('submission.csv' , 'wb') as fout:
        writer = csv.writer(fout)
        writer.writerow(['Id','Prediction'])
        for part_ind in xrange(4):
            print "Part ", part_ind
            f = open('backup/test_part{}pm.pckl'.format(part_ind), 'rb')
            prt = pickle.load(f)
            f.close()

            features = []
            alltext = []
            for ind, doc in enumerate(prt):
                doc_id, is_spam, url, words = doc
                alltext.append(words[1])
            doc_strs = []
            for i in alltext:
                doc_strs.append(" ".join(i))
            tfidfs = v.transform(doc_strs).toarray()
            features = []
            for ind, doc in enumerate(prt):
            #     print ind
                doc_id, is_spam, url, words = doc
                feature = calc_stats(words, url) + tfidfs[ind].tolist()
                features.append(feature)
            print len(prt)

            y = cl.predict(features)

            for i, doc in enumerate(prt):
                doc_id = doc[0]
                prediction = 1 if y[i] == True else 0
                writer.writerow([doc_id, prediction])
            del features, prt, alltext, doc_strs, tfidfs
