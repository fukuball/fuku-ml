#encoding=utf8

import os
import numpy as np
import FukuML.Utility as utility
import FukuML.SupportVectorMachine as svm

input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/emotion.dat')
input_other_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'FukuML/dataset/emotion_other.dat')

cross_validator = utility.CrossValidator()

svm_mc = svm.MultiClassifier()
svm_mc.load_train_data(input_train_data_file)
svm_mc.set_param(svm_kernel='soft_gaussian_kernel', gamma=1, C=9)
cross_validator.add_model(svm_mc)
avg_errors = cross_validator.excute()
print(avg_errors)

svm_mc = svm.MultiClassifier()
svm_mc.load_train_data(input_train_data_file)
svm_mc.load_test_data(input_other_data_file)
svm_mc.set_param(svm_kernel='soft_gaussian_kernel', gamma=1, C=9)
svm_mc.init_W()
svm_mc.train()

'''
for class_item in svm_mc.class_list:
    print(class_item)
    print(svm_mc.classifier_list[class_item].alpha)
    error_sv = (svm_mc.classifier_list[class_item].alpha > 0.999999999)
    print(np.arange(len(svm_mc.classifier_list[class_item].alpha))[error_sv])
'''

print("W 平均錯誤值（Ein）：")
print(svm_mc.calculate_avg_error_all_class(svm_mc.train_X, svm_mc.train_Y, svm_mc.W))
#print("W 平均錯誤值（Eout）：")
#print(svm_mc.calculate_avg_error_all_class(svm_mc.test_X, svm_mc.test_Y, svm_mc.W))

'''
data_num = len(svm_mc.train_Y)

for i in range(data_num):
    x_string = np.array(map(str, svm_mc.train_X[i]))
    x_string = ' '.join(x_string[1:])+' '+str(svm_mc.train_Y[i])
    prediction = svm_mc.prediction(x_string)
    if (float(prediction['prediction']) != float(prediction['input_data_y'])):
        print(i+1)
        print(prediction)
'''

data_num = len(svm_mc.test_Y)

for i in range(data_num):
    x_string = np.array(map(str, svm_mc.test_X[i]))
    x_string = ' '.join(x_string[1:])
    prediction = svm_mc.prediction(x_string, 'future_data')
    print(i+1)
    print(prediction)
