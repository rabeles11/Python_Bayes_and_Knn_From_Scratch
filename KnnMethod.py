import math

import numpy as np
import time

def resultsetofknntest(TestData, data, labels, k=3):
    resultset = []
    for PointOfTestData in TestData:
        soused_data, soused_labels = find_neighbours(PointOfTestData, data, labels, k)
        AnoCount = 0
        NeCounnt = 0
        for label in soused_labels:
            if label == "Yes":
                AnoCount += 1
            elif label == "No":
                NeCounnt += 1
        if AnoCount > NeCounnt:
            resultset.append("Yes")
        elif NeCounnt > AnoCount:
            resultset.append("No")
        else:
            resultset.append(soused_labels[0])
    return resultset


def find_neighbours(PointOfTestData, data, labels,
                    k=3):  # testovací jeden rádek, trénovací celá
    # Iterace skrze počet sousedu co má hledat! (tj budu opakovat vždy pro nejbližšího souseda, někam si ho uložit a pak ho vyhodit z toho datasetu, který projíždím aby jej našel)
    start_time = time.time()
    soused_data = []
    soused_labels = []
    for i in range(0, k):
        nejmensi_vzdalenost = 0
        index_nejblizsiho_prvku = 0
        for x in range(0, len(data)):  # tady od x ukládej vždy cislo řádku na kterém stojíš!! v trénovacích datech
            euklid_distance = 0
            for column in range(0, len(PointOfTestData)):  # tady ber čislo sloupce
                disctance = abs(PointOfTestData[column] - data[x][column])
                euklid_distance += disctance
            euklid_distance = math.sqrt(euklid_distance)
            # print(euklid_distance)

            if nejmensi_vzdalenost == 0:
                nejmensi_vzdalenost = euklid_distance
                index_nejblizsiho_prvku = x
            elif nejmensi_vzdalenost > euklid_distance:
                nejmensi_vzdalenost = euklid_distance
                index_nejblizsiho_prvku = x

                # tady počítat tu euklid vzdálenost
        soused_data.append(data[index_nejblizsiho_prvku])
        soused_labels.append(labels[index_nejblizsiho_prvku])

        data = np.delete(data, index_nejblizsiho_prvku, 0)
        labels = np.delete(labels, index_nejblizsiho_prvku, 0)
    print("--- %s seconds ---" % (time.time() - start_time))
    return soused_data, soused_labels

