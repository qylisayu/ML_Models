import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
import graphviz
import math

def load_data(REAL_TEXT_FILE_PATH, FAKE_TEXT_FILE_PATH):
    df_real = pd.read_csv(REAL_TEXT_FILE_PATH, header=None, names=["Text"])
    df_fake = pd.read_csv(FAKE_TEXT_FILE_PATH, header=None, names=["Text"])
    df_real["Type"] = 1
    df_fake["Type"] = 0
    df_entire = df_real.append(df_fake)
    df_x = df_entire["Text"]
    df_y = df_entire["Type"].values
    cv = CountVectorizer()
    x_traincv = cv.fit_transform(df_x)
    df_x_cv = x_traincv.toarray()
    X_train, X_test, y_train, y_test = train_test_split(df_x_cv, df_y, test_size=0.3, random_state=123)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=123)
    return (X_train, y_train, X_val, y_val, X_test, y_test, cv)

def select_model(X_train, y_train, X_val, y_val):
    columns_name=["max_depth", "split_criteria", "score"]
    criterion_list = ["gini", "entropy"]
    df = pd.DataFrame(columns=columns_name)
        
    for n in range(1, 50, 5):
        curr_max_depth = n
        
        for i in range(2):
            
            clf = tree.DecisionTreeClassifier(max_depth=curr_max_depth, criterion=criterion_list[i], random_state=123)
            clf = clf.fit(X_train, y_train)
            clf_score = clf.score(X_val, y_val, sample_weight=None)
        
            data = {columns_name[0]:curr_max_depth, columns_name[1]:criterion_list[i], columns_name[2]:clf_score}
            df = df.append(data, ignore_index=True)
        
    result = df.sort_values(by="score", ascending=False)
    result = result.reset_index(drop=True)
    print(result)
    
    depth = result["max_depth"][0]
    criteria_selected = result["split_criteria"][0]
    
    clf = tree.DecisionTreeClassifier(max_depth=depth, criterion=criteria_selected, random_state=123)
    clf = clf.fit(X_train, y_train)                
                
    return (clf, depth, criteria_selected)


def compute_information_gain(x_i, X_train , y_train, cv):
    df = pd.DataFrame(X_train)
    df["real/fake"] = y_train
    i = cv.get_feature_names().index(x_i)
  
    total_train = df.shape[0]
    num_real = df[df["real/fake"] == 1].shape[0]
    num_fake = df[df["real/fake"] == 0].shape[0]
    
  
    H_Y = - (num_real/total_train)*math.log(num_real/total_train, 2) - (num_fake/total_train)*math.log(num_fake/total_train, 2)
    
    word_real_perc= df[(df[i]>0.5) & (df["real/fake"] == 1)].shape[0]/total_train
    word_fake_perc = df[(df[i]>0.5) & (df["real/fake"] == 0)].shape[0]/total_train
    no_word_real_perc = df[(df[i]<0.5) & (df["real/fake"] == 1)].shape[0]/total_train
    no_word_fake_perc = df[(df[i]<0.5) & (df["real/fake"] == 0)].shape[0]/total_train
      
    H_Y_given_X = - word_real_perc * math.log(word_real_perc/(word_real_perc + word_fake_perc), 2) - word_fake_perc * math.log(word_fake_perc/(word_real_perc + word_fake_perc), 2) - no_word_real_perc * math.log(no_word_real_perc/(no_word_real_perc + no_word_fake_perc), 2) - no_word_fake_perc * math.log(no_word_fake_perc/(no_word_real_perc + no_word_fake_perc), 2)
    
    IG = H_Y - H_Y_given_X
    return IG
             

"""--------------------------------------------------------------------------------------------------------------"""

REAL_TEXT_FILE_PATH = '/Users/lisayu/Downloads/clean_real.txt'
FAKE_TEXT_FILE_PATH = '/Users/lisayu/Downloads/clean_fake.txt'

X_train, y_train, X_val, y_val, X_test, y_test, cv = load_data(REAL_TEXT_FILE_PATH, FAKE_TEXT_FILE_PATH)
clf, depth, criteria_selected = select_model(X_train, y_train, X_val, y_val)
IG_1 = compute_information_gain('the', X_train , y_train, cv)
IG_2 = compute_information_gain('hillary', X_train , y_train, cv)
IG_3 = compute_information_gain('trumps', X_train , y_train, cv)
IG_4 = compute_information_gain('donald', X_train , y_train, cv)

print('Information Gain of \'the\':' + str(IG_1))
print('Information Gain of \'hillary\':' + str(IG_2))
print('Information Gain of \'trumps\':' + str(IG_3))
print('Information Gain of \'donald\':' + str(IG_4))







