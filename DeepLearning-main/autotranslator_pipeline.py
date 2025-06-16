import matplotlib.pyplot as plt
import re
import numpy as np
import os
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split

print(f"TensorFlow Version: {tf.__version__}")

# Проверяем, существует ли файл с данными. Если нет - скачиваем.
path_to_file = 'rus.txt'
if not os.path.exists(path_to_file):
    print(f"Файл '{path_to_file}' не найден, скачиваю архив...")
    os.system('wget http://www.manythings.org/anki/rus-eng.zip')
    os.system('unzip rus-eng.zip')
else:
    print(f"Файл '{path_to_file}' найден, пропуск скачивания.")


def preprocess_sentence(w):
    """
    Функция предобработки предложения:
    - приводит к нижнему регистру, убирает лишние пробелы
    - отделяет знаки препинания пробелами
    - удаляет лишние символы
    - добавляет токены <start> и <end>
    """
    w = w.lower().strip()
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Zа-яА-Я?.!,]+", " ", w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w

def create_dataset(path, num_examples):
    """
    Читает пары предложений из файла.
    """
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]] for l in lines[:num_examples]]
    return zip(*word_pairs)

class LanguageIndex():
  """
  Класс для создания словарей (слово -> индекс и индекс -> слово).
  """
  def __init__(self, lang):
    self.lang = lang
    self.word2idx = {}
    self.idx2word = {}
    self.vocab = set()
    self.create_index()
    
  def create_index(self):
    vocab = set()
    for phrase in self.lang:
      vocab.update(phrase.split(' '))
    
    vocab = sorted(vocab)
    
    self.word2idx['<pad>'] = 0
    for index, word in enumerate(vocab):
      self.word2idx[word] = index + 1
    
    for word, index in self.word2idx.items():
      self.idx2word[index] = word

def load_dataset(path, num_examples, inp_lang_indexer, targ_lang_indexer):
    """
    Загрузка и подготовка датасета:
    - векторизация (текст в числа)
    - паддинг (выравнивание предложений до одной длины)
    """
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor = [[inp_lang_indexer.word2idx.get(s, 0) for s in sentence.split(' ')] for sentence in inp_lang]
    target_tensor = [[targ_lang_indexer.word2idx.get(s, 0) for s in sentence.split(' ')] for sentence in targ_lang]

    max_length_inp, max_length_targ = max(len(t) for t in input_tensor), max(len(t) for t in target_tensor)

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length_inp, padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=max_length_targ, padding='post')

    return input_tensor, target_tensor, max_length_inp, max_length_targ


def gru(units):
    """Вспомогательная функция для создания GRU слоя."""
    return tf.keras.layers.GRU(units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
    """
    Класс Энкодера (кодировщика).
    Принимает на вход предложение и сжимает его в вектор "смысла".
    """
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
    """
    Класс Декодера.
    Принимает вектор от энкодера и генерирует переведенное предложение.
    Использует механизм Attention.
    """
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # Слои для механизма внимания
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        x = self.fc(output)
        return x, state, attention_weights


# --- Гиперпараметры для обучения ---
num_examples = 30000
BUFFER_SIZE = 32000
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
EPOCHS = 10

def loss_function(real, pred):
    """
    Функция потерь. Игнорирует padding (<pad>) при расчете ошибки.
    """
    mask = tf.math.not_equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden, encoder, decoder, targ_lang_indexer, optimizer):
    """Один шаг обучения на одном батче."""
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang_indexer.word2idx['<start>']] * BATCH_SIZE, 1)

        # "Teacher forcing" - на вход декодеру подается правильное слово из таргета,
        # а не предсказанное на предыдущем шаге. Это ускоряет сходимость.
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

def train_model(input_tensor_train, target_tensor_train, encoder, decoder, targ_lang_indexer, epochs, checkpoint_prefix):
    """Основная функция обучения модели."""
    optimizer = tf.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    
    for epoch in range(epochs):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

        for (batch, (inp, targ)) in enumerate(dataset):
            batch_loss = train_step(inp, targ, enc_hidden, encoder, decoder, targ_lang_indexer, optimizer)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f'Эпоха {epoch+1} Батч {batch} Потери {batch_loss.numpy():.4f}')
        
        # Сохраняем модель (чекпоинт) каждые 2 эпохи
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Эпоха {epoch+1} Потери {total_loss / (len(input_tensor_train)//BATCH_SIZE):.4f}')
        print(f'Время на эпоху {time.time()-start:.2f} сек\n')

def evaluate(sentence, encoder, decoder, inp_lang_indexer, targ_lang_indexer, max_length_inp, max_length_targ):
    """
    Функция для перевода одного предложения (инференс/тестирование).
    """
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang_indexer.word2idx.get(i, 0) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang_indexer.word2idx['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        word = targ_lang_indexer.idx2word.get(predicted_id, '')
        
        if word == '<end>':
            return result, sentence
            
        result += word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)
        
    return result, sentence


# --- Обучение модели 1: Ru -> En ---
print("\n--- Обучение модели Ru -> En ---")

# Создаем словари
eng_lang, rus_lang = create_dataset(path_to_file, num_examples)
inp_lang_ru = LanguageIndex(rus_lang)
targ_lang_en = LanguageIndex(eng_lang)

# Загружаем датасет
input_tensor_ru, target_tensor_en, max_length_inp_ru, max_length_targ_en = load_dataset(
    path_to_file, num_examples, inp_lang_ru, targ_lang_en
)

# Делим данные на тренировочную и валидационную выборки
input_tensor_train_ru, _, target_tensor_train_en, _ = train_test_split(
    input_tensor_ru, target_tensor_en, test_size=0.2, random_state=42
)

# Создаем модели
encoder_ru_en = Encoder(len(inp_lang_ru.word2idx), embedding_dim, units, BATCH_SIZE)
decoder_ru_en = Decoder(len(targ_lang_en.word2idx), embedding_dim, units, BATCH_SIZE)

# Запускаем обучение
checkpoint_dir_ru_en = './training_checkpoints_ru_en'
checkpoint_prefix_ru_en = os.path.join(checkpoint_dir_ru_en, "ckpt")
train_model(input_tensor_train_ru, target_tensor_train_en, encoder_ru_en, decoder_ru_en, targ_lang_en, EPOCHS, checkpoint_prefix_ru_en)


# --- Обучение модели 2: En -> Ru ---
print("\n--- Обучение модели En -> Ru ---")

# Меняем местами входной и целевой языки
inp_lang_en = targ_lang_en
targ_lang_ru = inp_lang_ru

# Загружаем датасет
input_tensor_en, target_tensor_ru, max_length_inp_en, max_length_targ_ru = load_dataset(
    path_to_file, num_examples, inp_lang_en, targ_lang_ru
)

# Делим данные
input_tensor_train_en, _, target_tensor_train_ru, _ = train_test_split(
    input_tensor_en, target_tensor_ru, test_size=0.2, random_state=42
)

# Создаем второй, независимый набор моделей
encoder_en_ru = Encoder(len(inp_lang_en.word2idx), embedding_dim, units, BATCH_SIZE)
decoder_en_ru = Decoder(len(targ_lang_ru.word2idx), embedding_dim, units, BATCH_SIZE)

# Обучаем вторую модель
checkpoint_dir_en_ru = './training_checkpoints_en_ru'
checkpoint_prefix_en_ru = os.path.join(checkpoint_dir_en_ru, "ckpt")
train_model(input_tensor_train_en, target_tensor_train_ru, encoder_en_ru, decoder_en_ru, targ_lang_ru, EPOCHS, checkpoint_prefix_en_ru)


# --- Итоговый конвейер и оценка качества ---
def translate_rus_eng_rus(sentence):
    """
    Функция для полного цикла перевода: Ru -> En -> Ru.
    """
    print(f'\nИсходное предложение: {sentence}')
    
    # Шаг 1: Переводим с русского на английский
    intermediate_translation, _ = evaluate(sentence, encoder_ru_en, decoder_ru_en, inp_lang_ru, targ_lang_en, max_length_inp_ru, max_length_targ_en)
    print(f'Промежуточный перевод (En): {intermediate_translation}')
    
    # Шаг 2: Результат переводим с английского обратно на русский
    final_translation, _ = evaluate(intermediate_translation, encoder_en_ru, decoder_en_ru, inp_lang_en, targ_lang_ru, max_length_inp_en, max_length_targ_ru)
    print(f'Итоговый перевод (Ru): {final_translation}')
    
    return final_translation

# Запускаем оценку на тестовых примерах
translate_rus_eng_rus('я живу в липецке')
translate_rus_eng_rus('эта кошка умная') 