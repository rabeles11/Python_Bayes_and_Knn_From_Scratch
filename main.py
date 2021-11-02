import pandas as pd
from statistics import mean
from sklearn.metrics import classification_report, confusion_matrix

import BayesMethod
import KnnMethod

"""
Cesta k datasetu! -> Tohle změnit na vlastní dataset!
"""
df = pd.read_csv("C:\\Users\\9davi\\Downloads\\weatherAUS.csv")

"""
Začátek vytváření crosvalidačních datasetu
"""

df = df[df.columns.drop('Date').drop("Location").drop("WindGustDir").drop("WindDir9am").drop("WindDir3pm").drop(
    "RainToday")]
df_new = df.dropna().reset_index(drop=True)

DfRainTomorrowYes = df_new[df_new['RainTomorrow'] == 'Yes']
DfRainTomorrowNo = df_new[df_new['RainTomorrow'] == 'No']

NumberOfRows_DfRainTomorrowYes_10Per = round((len(DfRainTomorrowYes.index) / 100) * 10)
NumberOfRows_DfRainTomorrowNo_10Per = round((len(DfRainTomorrowNo.index) / 100) * 10)
start = 0
TestMnoziny = []
TrenMnoziny = []

start2 = 0
posouvac = NumberOfRows_DfRainTomorrowYes_10Per
posouvac2 = NumberOfRows_DfRainTomorrowNo_10Per
for x in range(10):
    pd.set_option("display.width", 5800)
    pd.set_option("display.max_columns", 30)
    first10percent = DfRainTomorrowYes[start:posouvac]
    second10percent = DfRainTomorrowNo[start2:posouvac2]
    start = posouvac
    posouvac += NumberOfRows_DfRainTomorrowYes_10Per
    start2 = posouvac2
    posouvac2 += NumberOfRows_DfRainTomorrowNo_10Per
    frames = [first10percent, second10percent]
    result = pd.concat(frames, axis=0, sort=False).sort_index()
    TestMnoziny.append(result)

for mnozina in TestMnoziny:
    df_drop = df_new
    training_result = pd.concat([df_drop, mnozina]).drop_duplicates(keep=False)
    TrenMnoziny.append(training_result)
"""
Konec vytváření crosvalidačních datasetu
"""

"""
Omezení vstupu do datasetu pro KNN -> pro přehledost
"""


def knn_30_results(dataset, od, do):
    X = TrenMnoziny[dataset].iloc[:, :-1].values
    y = TrenMnoziny[dataset].iloc[:, 17].values
    Xtest = TestMnoziny[dataset].iloc[:, :-1].values
    ytest = TestMnoziny[dataset].iloc[:, 17].values
    print("------------------------------" + "\n" + "Začátek metody KNN")
    vysledek = KnnMethod.resultsetofknntest(Xtest[od:do], X, y)
    print("Confusion matrix:")
    print(confusion_matrix(ytest[od:do], vysledek))
    creport = classification_report(ytest[od:do], vysledek, output_dict=True, zero_division=True)
    print("Přesnost Knn je : {} %".format(creport['accuracy'] * 100))
    print("Konec metody KNN" + "\n" + "------------------------------")


"""
Zde se volá metoda na Knn -> Číslo datasetu, začátek dat, konec dat -> tj -> 0,3010,3020 vezme první dataset a jeho
data od 3010 do 3020
"""
knn_30_results(0, 3010, 3020)

"""
Tady začíná výpočet Bayesmethod -> Tj zde se projíždí právě i daná crosvalidace tj všechny tyto datasety,
zde to lze udělat, protože to není tak časově náročné! 
"""
print("------------------------------" + "\n" + "Začátek metody Bayes")
list_of_probs = []
for x in range(len(TestMnoziny)):
    X = TrenMnoziny[x].iloc[:, :-1].values
    y = TrenMnoziny[x].iloc[:, 17].values
    Xtest = TestMnoziny[x].iloc[:, :-1].values
    ytest = TestMnoziny[x].iloc[:, 17].values
    Propability = BayesMethod.Bajes(TrenMnoziny[x], Xtest, ytest)
    list_of_probs.append(Propability)
valmean = mean(list_of_probs)
print("Výsledná Přesnost bayes metody je : {} %".format(valmean))
print("Konec metody Bayes" + "\n" + "------------------------------")
