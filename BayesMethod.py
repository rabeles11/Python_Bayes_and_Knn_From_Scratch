import math
from math import sqrt
from math import pi
from math import exp


def Bajes(TrenMnoziny,Xtest,ytest):
    groupeddataset = TrenMnoziny.groupby("RainTomorrow")
    AnosetMean = groupeddataset.get_group("Yes").mean(axis=0, numeric_only=True)
    NesetMean = groupeddataset.get_group("No").mean(axis=0, numeric_only=True)

    AnosetDeviation = groupeddataset.get_group("Yes").std(axis=0, numeric_only=True)
    NesetDeviation = groupeddataset.get_group("No").std(axis=0, numeric_only=True)
    listofprobs = []
    for x in range(0, len(Xtest) - 1):
        anopropability = 0
        nepropability = 0
        for i in range(0, len(Xtest[x]) - 1):
            try:
                anolog = math.log(calculate_probability(Xtest[x][i], AnosetMean[i], AnosetDeviation[i]))
            except:
                anolog = 0
            try:
                nelog = math.log(calculate_probability(Xtest[x][i], NesetMean[i], NesetDeviation[i]))
            except:
                nelog = 0
            anopropability = anopropability + anolog
            nepropability = nepropability + nelog
        if (anopropability > nepropability):
            listofprobs.append("Yes")
        else:
            listofprobs.append("No")

    print("{} %".format(countpropability(listofprobs, ytest)))
    return countpropability(listofprobs, ytest)


def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent


def countpropability(Result,Test):
    anotrefa = 0
    nespatne = 0
    for x in range(0,len(Result)):
        if(Result[x]==Test[x]):
            anotrefa += 1
        else:
            nespatne += 1
    procentualnitrefa = (anotrefa/len(Result)) * 100
    return procentualnitrefa