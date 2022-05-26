#Importing the required Dependencies
import tensorflow as tf
import numpy as np
from src.config import embedding_dim, units, vocab_size,BATCH_SIZE, top_k, model_path

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token='<unk>',
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
class BahdanauAttention(tf.keras.Model):
  def __init__(self,units):
    super(BahdanauAttention,self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V1 = tf.keras.layers.Dense(1)

  def call(self,features,hidden):
    hidden_with_time_axis = tf.expand_dims(hidden,1)
    score = tf.nn.tanh(self.W1(hidden_with_time_axis)+
                       self.W2(features))
    attention_weights = tf.nn.softmax(self.V1(score),axis=1)


    context_vector = attention_weights*features
    context_vector = tf.reduce_sum(context_vector,axis=1)
    return context_vector,attention_weights

class CNN_Encoder(tf.keras.Model):
  def __init__(self,embedding_dim):
    super(CNN_Encoder,self).__init__()
    self.fc = tf.keras.layers.Dense(embedding_dim)

  def call(self,x):
    x = self.fc(x)
    x = tf.nn.tanh(x)
    return x

class RNN_Decoder(tf.keras.Model):
  def __init__(self,embedding_dim,units,vocab_size):
    super(RNN_Decoder,self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim)
    self.GRU = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer = 'glorot_uniform')
    self.drop1 = tf.keras.layers.Dropout(0.4)
    self.batch_norm = tf.keras.layers.BatchNormalization()
    self.fc1 = tf.keras.layers.Dense(512)
    self.fc2 = tf.keras.layers.Dense(vocab_size)
    
    self.attention = BahdanauAttention(self.units)

  def call(self,x,features,hidden):
    context_vector,attention_weights = self.attention(features,hidden)
    x = self.embedding(x)

    x = tf.concat([tf.expand_dims(context_vector,1),x],axis=-1)

    output,state = self.GRU(x)
    x = self.fc1(output)
    x = self.drop1(x)
    x = self.batch_norm(x)
    x = tf.reshape(x, (-1, x.shape[2]))
    x = self.fc2(x)

    return x,state,attention_weights

  def reset_state(self,batch_sz):
    return tf.zeros([batch_sz,self.units])

def load_model():
    #Loss and Optimizer
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim,units,vocab_size)
    optimizer=tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
    #Initializing using Random Input
    test_x = encoder(np.ones([1,64,2048]))
    encoder.load_weights(f"{model_path}/weights1.hdf5")
    b = decoder.attention(test_x,decoder.reset_state(BATCH_SIZE))
    c = decoder(tf.expand_dims([tokenizer.word_index['<start>']]*BATCH_SIZE,1),test_x,decoder.reset_state(BATCH_SIZE))
    decoder.load_weights(f"{model_path}/weights2.hdf5")
    decoder.attention.load_weights(f"{model_path}/weights3.hdf5")
    return encoder, decoder