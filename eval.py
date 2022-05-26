#Importing the required Dependencies
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src.config import max_length, attention_features_shape
import io
import cv2

from conv_model import image_features_extract_model

def load_image(file_path):
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img,channels=3)
  img = tf.image.resize(img,(299,299))
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  #file_path = "./Pickle"+file_path[11:]
  print(file_path)
  return img,file_path

def evaluate(image, encoder, decoder, tokenizer):
  attention_plot = np.zeros([max_length,attention_features_shape])
  hidden = decoder.reset_state(1)
  temp_input = tf.expand_dims(load_image(image)[0], 0)
  img_tensor_val = image_features_extract_model(temp_input)
  img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

  features = encoder(img_tensor_val)
  dec_input = tf.expand_dims([tokenizer.word_index['<start>']],0)
  result = []
  for i in range(max_length):
    prediction,hidden,attention_weights = decoder(dec_input,features,hidden)
    attention_plot[i] = tf.reshape(attention_weights,(-1,)).numpy()
    predicted_id = tf.random.categorical(prediction,1)[0][0].numpy()
    result.append(tokenizer.index_word[predicted_id])
    if tokenizer.index_word[predicted_id] == '<end>':
        return result, attention_plot
    dec_input = tf.expand_dims([predicted_id],0)
  
  attention_plot = attention_plot[:len(result),:]
  return result,attention_plot


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def plot_attention(image,result,attention_plot):
  temp_image = np.array(Image.open(image))
  fig = plt.figure(figsize=(15,15))
  len_result = len(result)
  attention_plots = []
  for i in range(len_result):
    temp_att = np.resize(attention_plot[i],(8,8))
    ax =fig.add_subplot(len_result//2, len_result//2, i+1)
    ax.set_title(result[i])
    img = ax.imshow(temp_image)
    
    img = ax.imshow(temp_att,cmap='gray',alpha=0.6,extent=img.get_extent())
    attention_plots.append(img)


  return get_img_from_fig(fig, dpi=180)
 

