# 跟迴歸線比
def LSEerror(data, coef_array):
    error = 0
    for ele in data:
        new_y = 0
        for i in range(0, len(coef_array)):
            # 係數 * x次方（升次）
            new_y += coef_array[i][0] * pow(ele[0], i)
        # x在回歸線上對應y值 - 原y值
        error += pow(new_y - ele[1], 2)
    
    return error

# 兩個矩陣比
def LSEerrorMatrix(a, b):
    error = 0
    x = len(a)
    y = len(a[0])
    
    # 對應項相減取平方
    for i in range(0, x):
        for j in range(0, y):
            error += pow(a[i][j] - b[i][j] ,2)
            
    return error