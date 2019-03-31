from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from .keras import Layer
import unicodedata
import re
import numpy as np
import os
import time
from bs4 import BeautifulSoup
print(tf.__version__)


with tf.device('/device:GPU:0'):

        
    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
    
    def preprocess_sentence(w):
        w = unicode_to_ascii(w.lower().strip())
        
      
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        
  
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        
        w = w.rstrip().strip()
        
  
        w = '<start> ' + w + ' <end>'
        return w

    
    
    
    
     
    
    def create_dataset():
        y_raw = open("/content/drive/My Drive/europarl-v7.de-en.de", encoding='UTF-8').read().strip().split('\n')
        
    
    
    
    
    
        x_raw = open("/content/drive/My Drive/europarl-v7.de-en.en", encoding='UTF-8').read().strip().split('\n')
        
    
    
        lines=[]
        for l in range(int(0.2*len(x_raw))):
            temp=[x_raw[l],y_raw[l]]
            lines.append(str('\t'.join(temp)))
        
        word_pairs=[[preprocess_sentence(w) for w in l.split('\t')]  for l in lines]
        
        return word_pairs
    

    class LanguageIndex():
      def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        
        self.create_index()
        
      def create_index(self):
        for phrase in self.lang:
          self.vocab.update(phrase.split(' '))
        
        self.vocab = sorted(self.vocab)
        
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
          self.word2idx[word] = index + 1
        
        for word, index in self.word2idx.items():
          self.idx2word[index] = word
  
    def max_length(tensor):
        return max(len(t) for t in tensor)
    
    
    def load_dataset():
        
        pairs = create_dataset()
    
   
        inp_lang = LanguageIndex(en for en, de in pairs)
        targ_lang = LanguageIndex(de for en, de in pairs)
        
       
        
      
        input_tensor = [[inp_lang.word2idx[s] for s in en.split(' ')] for en, de in pairs]
        
      
        target_tensor = [[targ_lang.word2idx[s] for s in de.split(' ')] for en, de in pairs]
        
       
        max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
        
        input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                     maxlen=max_length_inp,
                                                                     padding='post')
        
        target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                      maxlen=max_length_tar, 
                                                                      padding='post')
        
        return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar
    
   
       
    input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset()
        
    
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
    
 
    
    
    
    
    
    
    
    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 100
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dim = 128
    units = 128
    vocab_inp_size = len(inp_lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)
    
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    
    
    def gru(units):
     
      if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units, 
                                        return_sequences=True, 
                                        return_state=True, 
                                        recurrent_initializer='glorot_uniform')
      else:
        return tf.keras.layers.GRU(units, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_activation='sigmoid', 
                                   recurrent_initializer='glorot_uniform')
        
        
        
    class Encoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
            super(Encoder, self).__init__()
            self.batch_sz = batch_sz
            self.enc_units = enc_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = gru(self.enc_units)
            
        def call(self, x, hidden):
            x = self.embedding(x)
            output, state = self.gru(x, initial_state = hidden)        
            return output, state
        
        def initialize_hidden_state(self):
            return tf.zeros((self.batch_sz, self.enc_units))
    
    
    
    class Decoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
            super(Decoder, self).__init__()
            self.batch_sz = batch_sz
            self.dec_units = dec_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = gru(self.dec_units)
            self.fc = tf.keras.layers.Dense(vocab_size)
            
            # used for attention
            self.W1 = tf.keras.layers.Dense(self.dec_units)
            self.W2 = tf.keras.layers.Dense(self.dec_units)
            self.V = tf.keras.layers.Dense(1)
        
        def call(self, x, hidden, enc_output):
         
            hidden_with_time_axis = tf.expand_dims(hidden, 1)
           
            
            Scaled-Dot attention
            score=self.V(tanh(self.W1(values)))
    		
    		
            attention_weights = tf.nn.softmax(score, axis=1)
       
            context_vector = attention_weights * enc_output
            context_vector = tf.reduce_sum(context_vector, axis=1)
            
            x = self.embedding(x)
            
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
            output, state = self.gru(x)
            
            output = tf.reshape(output, (-1, output.shape[2]))
          
            x = self.fc(output)
           
            return x, state, attention_weights
            
        def initialize_hidden_state(self):
            return tf.zeros((self.batch_sz, self.dec_units))
        
        
        
        
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    
    
    
    
    optimizer = tf.train.AdamOptimizer()
    
    
    def loss_function(real, pred):
      mask = 1 - np.equal(real, 0)
      loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
      return tf.reduce_mean(loss_)
    
    
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    
    
    ##Training
    
    EPOCHS = 5
    
    for epoch in range(EPOCHS):
        start = time.time()
        
        hidden = encoder.initialize_hidden_state()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                
                dec_hidden = enc_hidden
                
                dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)       
                
                
                for t in range(1, targ.shape[1]):
                   
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    
                    loss += loss_function(targ[:, t], predictions)
                    
                   
                    dec_input = tf.expand_dims(targ[:, t], 1)
            
            batch_loss = (loss / int(targ.shape[1]))
            
            total_loss += batch_loss
            
            variables = encoder.variables + decoder.variables
            
            gradients = tape.gradient(loss, variables)
            
            optimizer.apply_gradients(zip(gradients, variables))
            
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
       
        if (epoch + 1) % 2 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)
        
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    
    def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
        attention_plot = np.zeros((max_length_targ, max_length_inp))
        
        sentence = preprocess_sentence(sentence)
    
        inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
        inputs = tf.convert_to_tensor(inputs)
        
        result = ''
    
        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = encoder(inputs, hidden)
    
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)
    
        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
            
         
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()
    
            predicted_id = tf.argmax(predictions[0]).numpy()
    
            result += targ_lang.idx2word[predicted_id] + ' '
    
            if targ_lang.idx2word[predicted_id] == '<end>':
                return result, sentence, attention_plot
            
          
            dec_input = tf.expand_dims([predicted_id], 0)
    
        return result, sentence, attention_plot
    
    
  
    
    
    
    
    
    def translate(sentence, actual, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
        result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
            
#      
        bleu_score=(sentence_bleu(actual, result, weights=(1, 0, 0, 0))+sentence_bleu(actual, result, weights=(0.5, 0.5, 0, 0))+sentence_bleu(actual, result, weights=(0.33, 0.33, 0.33, 0))+sentence_bleu(actual, result, weights=(0.33, 0.33, 0.33, 0)))/4
     
        return bleu_score,result
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

 ##Testing & Evaluation
    test_en_input = BeautifulSoup(open("newstest2014-deen-ref.en.sgm"), 'html.parser')
    test_de_ref=BeautifulSoup(open("newstest2014-deen-ref.de.sgm"), 'html.parser')
    text_en=test_en_input.get_text()
    lines_en=text_en.split('\n')
    text_de=test_de_ref.get_text()
    lines_de=text_de.split('\n')
    
    
    input_en=[]
    ref_de=[]
    for item in lines_en:
        if len(item)!=0:
            input_en.append(item)
    for item in lines_de:
        if len(item)!=0:
            ref_de.append(item)
    BLEU=0;        
    for l,m in input_en,ref_de:
        _,score=translate(l,m,encoder,decoder,inp_lang,targ_lang,max_length_inp,max_length_targ);
        BLEU=BLEU+score*100
        
    print("Overall BLEU score on Test Data is : %f",BLEU)
   
    #Printing Translations from Test Data
    
    m=[17,78,81]
    
    for i in m:
        result,_=translate(l,m,encoder,decoder,inp_lang,targ_lang,max_length_inp,max_length_targ)
        print("Input Sentence:   ",format(input_en[i]))
        print("Predicted Sentence:   ",format(result))
    
        
    