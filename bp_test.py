# coding:UTF-8
'''
Date:20160831
@author: zhaozhiyong
'''
import numpy as np
from bp_train import get_predict
def load_data(file_name):
    # 1、获取特征
    f = open(file_name)  # 打开文件
    feature_data = []
    label_tmp = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split(",")
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(int(lines[-1]))
        feature_data.append(feature_tmp)
    f.close()  # 关闭文件
    # 2、获取标签
    m = len(label_tmp)
    n_class = len(set(label_tmp))  # 得到类别的个数
    label_data = np.mat(np.zeros((m, n_class)))
    for i in range(m):
        label_data[i, label_tmp[i]] = 1
    return np.mat(feature_data), label_data, n_class

def load_model(file_w0, file_w1, file_b0, file_b1):    
    def get_model(file_name):
        f = open(file_name)
        model = []
        for line in f.readlines():
            lines = line.strip().split("\t")
            model_tmp = []
            for x in lines:
                model_tmp.append(float(x.strip()))
            model.append(model_tmp)
        f.close()
        return np.mat(model)   
    w0 = get_model(file_w0)
    w1 = get_model(file_w1)
    b0 = get_model(file_b0)
    b1 = get_model(file_b1)
    return w0, w1, b0, b1
def save_predict(file_name, pre):
    f = open(file_name, "w")
    m = np.shape(pre)[0]
    result = []
    for i in range(m):
        result.append(str(pre[i, 0]))
    f.write("\n".join(result))
    f.close()
def err_rate(label, pre):
    m = np.shape(label)[0]
    err = 0.0
    for i in range(m):
        if label[i, 0] != pre[i, 0]:
            err += 1
    rate = err / m
    return rate
if __name__ == "__main__":
    # 1、导入测试数据
    print("--------- 1.load data ------------")
    dataTest,label,n_class = load_data("test_data.csv")
    # 2、导入BP神经网络模型
    print("--------- 2.load model ------------")
    w0, w1, b0, b1 = load_model("weight_w0", "weight_w1", "weight_b0", "weight_b1")
    # 3、得到最终的预测值
    print("--------- 3.get prediction ------------")
    result = get_predict(dataTest, w0, w1, b0, b1)
    # 4、保存最终的预测结果
    print("--------- 4.save result ------------")
    pre = np.argmax(result, axis=1)
    save_predict("result", pre)
    print("预测准确性为：", (1 - err_rate(np.argmax(label, axis=1), np.argmax(result, axis=1))))