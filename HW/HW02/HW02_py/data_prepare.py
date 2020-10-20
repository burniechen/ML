import numpy as np

def GetImageData(f):
    # magic number
    f.read(4) 
    
    # number of images
    num = f.read(4)
    num = int.from_bytes(num, byteorder='big') #60000

    row = f.read(4)
    row = int.from_bytes(row, byteorder='big') #28

    column = f.read(4)
    column = int.from_bytes(column, byteorder='big') #28

    buf = f.read(row * column * num)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num, row * column)
    
    return [data, num, row, column]

def GetLabelData(f):
    # magic number
    f.read(4) 

    # number of items
    num = f.read(4)
    num = int.from_bytes(num, byteorder='big') #10000

    buf = f.read(num)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num)
    
    return [data, num]

def GetIndexOfEachLabel(label_data):
    index = 0
    index_m = []
    for num in range(10):
        index_m.append([])
        for label in label_data:
            if label == num:
                index_m[num].append(index)
            index +=1
        index = 0
        
    return index_m