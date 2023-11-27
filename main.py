# import modules
import pandas as pd
import numpy as np
import math
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from statistics import mode
from statistics import StatisticsError

# Split data
def split(initial_data, change, label_selection):
    # 將資料分為前半(first_half)與後半(second_half)
    split_data = np.split(initial_data, 6)
    first_half = [split_data[0], split_data[2], split_data[4]]
    second_half = [split_data[1], split_data[3], split_data[5]]
    
    '''select data'''
    if change:              # 前半資料為training data
        if label_selection[0] & label_selection[1]:
            training_data = np.vstack((first_half[0], first_half[1]))
            test_data = np.vstack((second_half[0], second_half[1]))
        elif label_selection[1] & label_selection[2]:
            training_data = np.vstack((first_half[1], first_half[2]))
            test_data = np.vstack((second_half[1], second_half[2]))
        else:
            training_data = np.vstack((first_half[0], first_half[2]))
            test_data = np.vstack((second_half[0], second_half[2]))
    else:                   # 後半資料為training data
        if label_selection[0] & label_selection[1]:
            training_data = np.vstack((second_half[0], second_half[1]))
            test_data = np.vstack((first_half[0], first_half[1]))
        elif label_selection[1] & label_selection[2]:
            training_data = np.vstack((second_half[1], second_half[2]))
            test_data = np.vstack((first_half[1], first_half[2]))
        else:
            training_data = np.vstack((second_half[0], second_half[2]))
            test_data = np.vstack((first_half[0], first_half[2]))

    return training_data, test_data
                
# calculate weight vector及bias
def get_w_b(data, feature_1, feature_2, c, class_P, class_N):
    # 根據要使用的特徵決定feature vector
    x = data[:, (feature_1 - 1):feature_2]
    
    # label
    y = data[:, 4]              # class label
    # print(y)
    # 分割為正類別與負類別
    x_P = []
    x_N = []
    y_P = []
    y_N = []
    for i in range(len(x)):
        if y[i] == class_P:
            x_P.append(x[i])
            y_P.append(y[i])
        else:
            x_N.append(x[i])
            y_N.append(y[i])
    
    n1 = len(x_P)                       # 正類別資料數量
    n2 = len(x_N)                       # 負類別資料數量
    # print(x_P)

    '''計算平均向量'''
    # 正類別
    m1 = np.zeros(shape = len(x_P[0]))
    for i in range(len(m1)):
        for j in range(len(x_P)):
            m1[i] += x_P[j][i]
        
    for k in range(len(m1)):
        m1[k] = round((m1[k] / n1), 4)
    m1 = np.array([m1])
    # print(m1)

    # 負類別
    m2 = np.zeros(shape = len(x_N[0]))
    for i in range(len(m2)):
        for j in range(len(x_N)):
            m2[i] += x_N[j][i]
    
    for k in range(len(m2)):
        m2[k] = round((m2[k] / n2), 4)
    m2 = np.array([m2])
    # print('m1 = ', m1)
    # print('m2 = ', m2)

    '''prior probability'''
    p1 = n1/(n1 + n2)           # 正類別先驗機率
    p2 = n2/(n1 + n2)           # 負類別先驗機率
    
    '''covariance matrix'''
    # 正類別
    co_P = np.zeros(shape = len(m1) * len(m1))
    for i in range(len(x_P)):
        # print(((np.array(x_P[i]) - m1).T).dot(np.array(x_P[i]) - m1))
        co_P = co_P + ((x_P[i] - m1).T).dot(x_P[i] - m1)
        # print(co_P)
    covariance1 = (1/(n1 - 1)) * co_P
    # print(co_P)

    # 負類別
    co_N = np.zeros(shape = len(m2) * len(m2))
    for i in range(len(x_N)):
        co_N = co_N + ((x_N[i] - m2).T).dot(x_N[i] - m2)
    covariance2 = (1/(n2 - 1)) * co_N

    # print(covariance1)
    # print(covariance2)
    covariance = p1*covariance1 + p2*covariance2
    # print(covariance)

    '''weight vector'''
    covariance_inv = np.linalg.inv(covariance)

    w = (m1 - m2).dot(covariance_inv)
    # 取小數點下第五位作為return的值
    for i in range(len(w[0])):
        w[0][i] = round(w[0][i], 5)
    # print("w = ", w)

    '''bias'''
    b = (-1/2) * (m1 - m2).dot(covariance_inv).dot((m1 + m2).T) - math.log((c*(p1/p2)))
    # 取小數點下第五位作為return的值
    b = round(float(b[0][0]), 5)
    # print("b = ", b)

    return w, b


# Decision function
def Decision(test_data, feature_1, feature_2, w, b, class_P, class_N):
    # 根據要使用的特徵決定feature vector
    x = test_data[:, (feature_1 - 1):feature_2]

    '''Decision function'''
    # 存儲最後預測所得的label
    predict = []
    
    for i in range(len(x)):
        D = w.dot(x[i]) + b
        if D[0] > 0:
            predict.append(class_P)
        elif D[0] < 0:
            predict.append(class_N)
        else:
            predict.append("false")
    
    return predict


# 計算分類率
def classification_rate(test_data, predict):
    # 預測正確的資料總數
    True_prediction = 0

    # 將predict的label與test data的label做比對
    for i in range(len(predict)):
        if predict[i] == test_data[i][4]:
            True_prediction += 1
    
    # 分類率
    # print(True_prediction)
    CR = round(True_prediction / len(test_data), 5) * 100
    return CR


# 更動training data與test data順序並回傳結果
def LDA(initial_data, feature1, feature2):
    # label為class 2, 3
    label_selection = [0, 1, 1]

    '''前半資料為training data'''
    change = 1
    c = 1
    # split data
    training_data_1, test_data_1 = split(initial_data, change, label_selection)
    # for i in range(len(test_data_1)):
    #     if i < (len(test_data_1)/2):
    #         training_data_1[i][4] = 1
    #         test_data_1[i][4] = 1
    #     else:
    #         training_data_1[i][4] = -1
    #         test_data_1[i][4] = -1
    
    # Decision function
    w_1, b_1 = get_w_b(training_data_1, feature1, feature2, c, class_P=2, class_N=3)
    # Prediction
    predict1 = Decision(test_data_1, feature1, feature2, w_1, b_1, class_P=2, class_N=3)
    CR1 = classification_rate(test_data_1, predict1)


    '''後半資料為training data'''
    change = 0
    # split data
    training_data_2, test_data_2 = split(initial_data, change, label_selection)
    
    # Decision function
    w_2, b_2 = get_w_b(training_data_2, feature1, feature2, c, class_P=2, class_N=3)
    # Prediction
    predict2 = Decision(test_data_2, feature1, feature2, w_2, b_2, class_P=2, class_N=3)
    CR2 = classification_rate(test_data_2, predict2)

    Average_CR = round((CR1 + CR2)/2, 2)       # Average CR

    # 輸出至file
    with open(f"LDA_class23_特徵{feature1}-{feature2}.csv", "a", newline="") as file:
        file.write("前半資料為training data\n")
    output_result(test_data_1, feature1, feature2, predict1, CR1, w_1, b_1)

    with open(f"LDA_class23_特徵{feature1}-{feature2}.csv", "a", newline="") as file:
        file.write("後半資料為training data\n")
    output_result(test_data_2, feature1, feature2, predict2, CR2, w_2, b_2)
    with open(f"LDA_class23_特徵{feature1}-{feature2}.csv", "a", newline="") as file:
        file.write(f"Average CR = {'%.2f'%round(Average_CR, 2)} %\n\n")


# 多類別分類
def LDA_mult(initial_data, feature1, feature2):
    # 儲存預測結果
    predict_set = []

    # 多數決後的預測結果
    predict_final = []

    # 儲存各種組合的w跟b
    w_set = []
    b_set = []

    change = 0
    c = 1

    # test_data共有75筆資料
    split_data = np.split(initial_data, 6)
    first_half = [split_data[0], split_data[2], split_data[4]]
    second_half = [split_data[1], split_data[3], split_data[5]]
    if change:
        test_data = np.vstack((second_half[0], second_half[1], second_half[2]))
    else:
        test_data = np.vstack((first_half[0], first_half[1], first_half[2]))
    

    class_set = [[1, 2], [1, 3], [2, 3]]
    for i in range(len(class_set)):
        label_selection = [0, 0, 0]
        class_1 = class_set[i][0]               # 正類別
        class_2 = class_set[i][1]               # 負類別
        label_selection[class_1 - 1] = 1
        label_selection[class_2 - 1] = 1

        # split data
        training_data, _ = split(initial_data, change, label_selection)

        # Decision function
        w, b = get_w_b(training_data, feature1, feature2, c, class_1, class_2)
        w_set.append(w)
        b_set.append(b)

        # Prediction
        predict_set.append(Decision(test_data, feature1, feature2, w_set[i], b_set[i], class_1, class_2))
        
    predict_set = np.array(predict_set)

    for j in range(len(predict_set.T)):
        try:
            find_mode = mode(predict_set.T[j])
            # print("predict = ", find_mode)
            predict_final.append(find_mode)
        except StatisticsError:
            # 如果 mode() 引發错误，表示所有元素數量相同
            # print("Error")
            predict_final.append("Error")    # 判定為分類錯誤
            continue
    
    # print(predict_final)
    # print(test_data[:, 4])
    CR = classification_rate(test_data, predict_final)
    # print(CR)

    with open(f"多類別分類_特徵{feature1}-{feature2}.csv", "a", newline="") as file:
        if change:
            file.write("前半資料為training data\n")
        else:
            file.write("後半資料為training data\n")
        
        for i in range(len(class_set)):
            file.write(f"class = , {class_set[i][0]}, {class_set[i][1]}\n")
            for j in range(len(predict_set[i])):
                file.write(f"{int(predict_set[i][j])}, ")
            file.write("\n")
            file.write(f"weight vector = , {w_set[i][0][0]}, {w_set[i][0][1]}\n")
            file.write(f"bias = , {b_set[i]}\n")
        
        file.write("\npredict_final = \n")
        for i in predict_final:
            file.write(f"{i}, ")

        file.write(f"\nCR = {'%.2f'%round(CR, 2)} %\n\n")


# 計算ROC與AUC
def LDA_ROC_AUC(initial_data, feature1, feature2):
    total_FPR_TPR = []                  # change=0,1時的FPR跟TPR
    total_AUC_score = []                # change=0,1時的AUC score
    avg_FPR_TPR = []                    # 平均的FPR跟TPR
    avg_AUC_score = []                  # 平均的AUC score
    for i in range(2):
        # change=0表示後半資料為training data
        change = i
        # print("change = ", change)
        
        with open(f"ROC_AUC_特徵{feature1}-{feature2}.csv", "a", newline="") as file:
            if change:
                file.write("前半資料為training data\n")
            else:
                file.write("後半資料為training data\n")

        # label為class 2, 3
        label_selection = [0, 1, 1]

        # split data
        training_data, test_data = split(initial_data, change, label_selection)

        # class 3為positive; class 2為negative
        class_2, class_3 = np.split(test_data, 2)
        test_data = np.vstack((class_3, class_2))
        # print(label_test_data)

        w_set = []              # weight vector
        b_set = []              # bias
        predict_set = []        # prediction
        FPR_TPR_set = []        # store FPR and TPR
        AUC_set = []            # store AUC score for each c

        for j in range(1, 10001):
            c = j/10000

            # Decision function
            w, b = get_w_b(training_data, feature1, feature2, c, class_P=3, class_N=2)
            w_set.append(w)
            b_set.append(b)

            # Prediction
            predict = Decision(test_data, feature1, feature2, w, b, class_P=3, class_N=2)
            predict_set.append(predict)

            # confusion matrix
            cf_mat = confusion_matrix(test_data[:, 4], predict)
            # cf_mat = [[TN, FP],
            #           [FN, TP]]

            TPR = cf_mat[1][1] / (cf_mat[1][1] + cf_mat[1][0])
            FPR = cf_mat[0][1] / (cf_mat[0][1] + cf_mat[0][0])
            FPR_TPR_set.append([FPR, TPR])

            # calculate AUC score
            auc = roc_auc_score(test_data[:, 4], predict) / 10000
            # print("auc = ", auc)
            AUC_set.append(round(auc, 5))

            with open(f"ROC_AUC_特徵{feature1}-{feature2}.csv", "a", newline="") as file:
                file.write(f"c1/c2 = {c}\n")
                file.write(f"weight vector = , {w[0][0]}, {w[0][1]}\n")
                file.write(f"bias = , {b}\n")
                file.write(f"FPR = {FPR}, TPR = {TPR}\n")
                file.write(f"AUC score = {auc}\n\n")
        
        total_FPR_TPR.append(FPR_TPR_set)
        total_AUC_score.append(AUC_set)

    # print(total_AUC_score)
    
    for i in range(len(total_FPR_TPR[0])):
        avg_FPR = round(((total_FPR_TPR[0][i][0] + total_FPR_TPR[1][i][0]) / 2), 2)
        avg_TPR = round(((total_FPR_TPR[0][i][1] + total_FPR_TPR[1][i][1]) / 2), 2)

        avg_auc = round(((total_AUC_score[0][i] + total_AUC_score[1][i]) / 2), 10)

        avg_FPR_TPR.append([avg_FPR, avg_TPR])
        avg_AUC_score.append(avg_auc)

    return avg_FPR_TPR, avg_AUC_score


# 將結果輸出為csv
def output_result(test_data, feature1, feature2, predict, CR, w, b):
    with open(f"LDA_class23_特徵{feature1}-{feature2}.csv", "a", newline="") as file:
        file.write("test data = \n")
        for i in test_data:
            file.write(f"{int(i[4])}, ")
        file.write(f"\n取特徵{feature1} {feature2}\n")

        # 將weight vector及bias輸出為csv
        # 紀錄的值為取小數點下第二位並保留末尾的0
        file.write(f"weight vector = [{'%.2f'%round(float(w[0][0]), 2)}  {'%.2f'%round(float(w[0][1]), 2)}]\n")
        file.write("bias = ")
        file.write(f"{'%.2f'%round(b, 2)}\n")

        file.write("predict result = \n")
        for i in predict:
            file.write(f"{i}, ")
        file.write(f"\nCR = {'%.2f'%round(CR, 2)} %\n\n")
    

# 畫ROC curve
def plot_ROC_curve(avg_FPR_TPR, avg_AUC_score):
    auc = 0
    for i in range(len(avg_AUC_score)):
        auc += round(avg_AUC_score[i], 10)
    auc = round(auc, 2)
    avg_FPR_TPR = np.array(avg_FPR_TPR)

    plt.figure()
    plt.plot(avg_FPR_TPR.T[0], avg_FPR_TPR.T[1], label="ROC (AUC = %0.2f)" % auc, lw=2)
    plt.xlim(0.0, 1.05)
    plt.ylim(0.0, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    
    
def main():
    # load data
    initial_data = np.loadtxt('iris.txt', dtype=float)

    '''Parameter settings'''

    # 要選取的class
    class_1 = 2
    class_2 = 3

    # 選擇要使用的特徵範圍
    feature1 = 3
    feature2 = 4

    '''LDA 作業part1'''
    LDA(initial_data, feature1, feature2)

    '''LDA 作業part2'''
    LDA_mult(initial_data, feature1, feature2)

    '''LDA 作業part3'''
    avg_FPR_TPR, avg_AUC_score = LDA_ROC_AUC(initial_data, feature1, feature2)
    plot_ROC_curve(avg_FPR_TPR, avg_AUC_score)

if __name__ == "__main__":
    main()