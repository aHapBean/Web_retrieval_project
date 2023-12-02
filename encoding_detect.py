import chardet
# 检测文件编码，认为是
file_path = 'training.1600000.processed.noemoticon.csv'
chunk_size = 102400
with open(file_path, 'rb') as file:
    # 读取文件内容
    content = file.read(chunk_size)

# 使用chardet检测编码
result = chardet.detect(content)

# 输出检测结果
print(f"The detected encoding is: {result['encoding']}, with confidence {result['confidence']:.2f}")
