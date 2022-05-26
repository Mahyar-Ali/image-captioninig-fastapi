BATCH_SIZE = 64
BUFFER_SIZE=1000
embedding_dim = 256
units=512
top_k = 10000
vocab_size = top_k+1
'''Shape of the vector extracted from InceptionV3 is (64, 2048)'''
features_shape = 2048
attention_features_shape = 64
max_length = 49
model_path = 'model'