import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import re
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split

# --- Инициализация ---
print(f"PyTorch Version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# --- Загрузка и предварительная обработка данных ---

path_to_file = 'rus.txt'
if not os.path.exists(path_to_file):
    print(f"Файл '{path_to_file}' не найден, скачиваю архив...")
    os.system('wget http://www.manythings.org/anki/rus-eng.zip')
    os.system('unzip rus-eng.zip')
else:
    print(f"Файл '{path_to_file}' найден, пропуск скачивания.")


def preprocess_sentence(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Zа-яА-Я?.!,]+", " ", w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w

def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]] for l in lines[:num_examples]]
    return zip(*word_pairs)

class LanguageIndex():
  def __init__(self, lang):
    self.lang = lang
    self.word2idx = {}
    self.idx2word = {}
    self.create_index()
    
  def create_index(self):
    # Жестко задаем индексы для специальных токенов
    self.word2idx = {'<pad>': 0, '<unk>': 1}
    self.idx2word = {0: '<pad>', 1: '<unk>'}
    
    # Собираем уникальные слова из корпуса
    vocab = set()
    for phrase in self.lang:
      vocab.update(phrase.split(' '))
    
    # Добавляем остальные слова в словарь
    idx = 2
    for word in sorted(list(vocab)):
      if word not in self.word2idx:
        self.word2idx[word] = idx
        self.idx2word[idx] = word
        idx += 1

def pad_sequences(sequences, maxlen, padding='post', value=0):
    padded_sequences = np.full((len(sequences), maxlen), value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq = seq[:maxlen]
        if padding == 'post':
            padded_sequences[i, :len(seq)] = seq
        else:
            padded_sequences[i, -len(seq):] = seq
    return padded_sequences

def load_dataset(path, num_examples, inp_lang_indexer, targ_lang_indexer):
    targ_lang, inp_lang = create_dataset(path, num_examples)

    # Получаем индексы для неизвестных слов
    unk_idx_inp = inp_lang_indexer.word2idx['<unk>']
    unk_idx_targ = targ_lang_indexer.word2idx['<unk>']

    input_tensor = [[inp_lang_indexer.word2idx.get(s, unk_idx_inp) for s in sentence.split(' ')] for sentence in inp_lang]
    target_tensor = [[targ_lang_indexer.word2idx.get(s, unk_idx_targ) for s in sentence.split(' ')] for sentence in targ_lang]

    max_length_inp, max_length_targ = max(len(t) for t in input_tensor), max(len(t) for t in target_tensor)

    # Используем правильный индекс для паддинга
    pad_idx_inp = inp_lang_indexer.word2idx['<pad>']
    pad_idx_targ = targ_lang_indexer.word2idx['<pad>']

    input_tensor = pad_sequences(input_tensor, maxlen=max_length_inp, padding='post', value=pad_idx_inp)
    target_tensor = pad_sequences(target_tensor, maxlen=max_length_targ, padding='post', value=pad_idx_targ)

    return input_tensor, target_tensor, max_length_inp, max_length_targ


# --- Модели на PyTorch ---

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Encoder, self).__init__()
        self.hidden_units = hidden_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_units, batch_first=True)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.gru(x, hidden)
        return output, hidden

    def initialize_hidden_state(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_units), device=device)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim + hidden_units, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, vocab_size)

        # Attention
        self.W1 = nn.Linear(hidden_units, hidden_units)
        self.W2 = nn.Linear(hidden_units, hidden_units)
        self.V = nn.Linear(hidden_units, 1)

    def forward(self, x, hidden, enc_output):
        hidden_with_time_axis = hidden.permute(1, 0, 2)
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        attention_weights = F.softmax(self.V(score), dim=1)

        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)

        x = self.embedding(x)
        x = torch.cat((context_vector.unsqueeze(1), x), -1)

        output, state = self.gru(x, hidden)
        output = output.squeeze(1)
        prediction = self.fc(output)
        
        return prediction, state, attention_weights


# --- Цикл Обучения ---

def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, targ_lang_indexer, batch_size):
    encoder_hidden = encoder.initialize_hidden_state(batch_size)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([[targ_lang_indexer.word2idx['<start>']]] * batch_size, device=device, dtype=torch.long)
    
    total_loss = 0.0
    total_tokens = 0  # количество НЕ-паддингов, учтённых в потере

    pad_idx = targ_lang_indexer.word2idx['<pad>']

    for t in range(1, target_tensor.shape[1]):
        predictions, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)

        # Считаем только те позиции, где целевой токен не PAD
        non_pad_mask = target_tensor[:, t] != pad_idx
        if non_pad_mask.any():
            step_loss = criterion(predictions[non_pad_mask], target_tensor[:, t][non_pad_mask])
            total_loss += step_loss
            total_tokens += non_pad_mask.sum().item()

        # Teacher forcing
        decoder_input = target_tensor[:, t].unsqueeze(1)

    # Нормируем на количество реально учтённых токенов, чтобы получить усреднённую потерю
    avg_loss = total_loss / max(total_tokens, 1)

    avg_loss.backward()

    # Обрезка градиента для предотвращения "взрыва"
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return avg_loss.item()

def train_model(input_tensor, target_tensor, encoder, decoder, targ_lang_indexer, epochs, batch_size, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    # Игнорируем пад-токены и суммируем лог-loss, чтобы избежать NaN при отсутствии валидных токенов
    criterion = nn.CrossEntropyLoss(ignore_index=targ_lang_indexer.word2idx['<pad>'], reduction='sum')
    
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(input_tensor), torch.from_numpy(target_tensor))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        
        encoder.train()
        decoder.train()

        for batch, (inp, targ) in enumerate(dataloader):
            inp, targ = inp.to(device), targ.to(device)
            
            batch_loss = train_step(inp, targ, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, targ_lang_indexer, batch_size)
            
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f'Эпоха {epoch+1} Батч {batch} Потери {batch_loss:.4f}')
        
        if (epoch + 1) % 2 == 0:
            torch.save(encoder.state_dict(), os.path.join(checkpoint_path, f'encoder_epoch_{epoch+1}.pth'))
            torch.save(decoder.state_dict(), os.path.join(checkpoint_path, f'decoder_epoch_{epoch+1}.pth'))

        print(f'Эпоха {epoch+1} Потери {total_loss / len(dataloader):.4f}')
        print(f'Время на эпоху {time.time()-start:.2f} сек\n')


# --- Оценка модели ---

def evaluate(sentence, encoder, decoder, inp_lang_indexer, targ_lang_indexer, max_length_inp, max_length_targ):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        sentence = preprocess_sentence(sentence)
        inputs = [inp_lang_indexer.word2idx.get(i, 0) for i in sentence.split(' ')]
        inputs = pad_sequences([inputs], maxlen=max_length_inp, padding='post', value=inp_lang_indexer.word2idx['<pad>'])
        inputs = torch.from_numpy(inputs).to(device)
        
        result = ''
        
        encoder_hidden = encoder.initialize_hidden_state(batch_size=1)
        enc_out, enc_hidden = encoder(inputs, encoder_hidden)

        dec_hidden = enc_hidden
        dec_input = torch.tensor([[targ_lang_indexer.word2idx['<start>']]], device=device)

        for _ in range(max_length_targ):
            predictions, dec_hidden, attn_weights = decoder(dec_input, dec_hidden, enc_out)

            predicted_id = torch.argmax(predictions[0]).item()
            word = targ_lang_indexer.idx2word.get(predicted_id, '')

            # Если модель предсказала токен <unk>, попробуем "скопировать" наиболее релевантное слово из входа
            if word == '<unk>':
                # attn_weights: [batch, seq_len, 1] -> возьмём seq_len
                attn_vector = attn_weights.squeeze(-1)[0]  # shape: (seq_len,)
                # индекс токена входного предложения с максимальным весом внимания
                src_token_idx = torch.argmax(attn_vector).item()
                copied_word_idx = inputs[0, src_token_idx].item()
                word = inp_lang_indexer.idx2word.get(copied_word_idx, '')

            if word == '<end>':
                return result.strip(), sentence

            result += word + ' '
            dec_input = torch.tensor([[predicted_id]], device=device)
            
    return result.strip(), sentence

# --- Основной скрипт ---

# Гиперпараметры
num_examples = 30000
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
EPOCHS = 10

# --- Модель 1: Ru -> En ---
print("\n--- Обучение модели Ru -> En (PyTorch) ---")
eng_lang, rus_lang = create_dataset(path_to_file, num_examples)
inp_lang_ru = LanguageIndex(rus_lang)
targ_lang_en = LanguageIndex(eng_lang)

input_tensor_ru, target_tensor_en, max_length_inp_ru, max_length_targ_en = load_dataset(
    path_to_file, num_examples, inp_lang_ru, targ_lang_en)
input_train_ru, _, target_train_en, _ = train_test_split(
    input_tensor_ru, target_tensor_en, test_size=0.2, random_state=42)

encoder_ru_en = Encoder(len(inp_lang_ru.word2idx), embedding_dim, units).to(device)
decoder_ru_en = Decoder(len(targ_lang_en.word2idx), embedding_dim, units).to(device)

checkpoint_dir_ru_en = './training_checkpoints_ru_en_pytorch'
train_model(input_train_ru, target_train_en, encoder_ru_en, decoder_ru_en, targ_lang_en, EPOCHS, BATCH_SIZE, checkpoint_dir_ru_en)


# --- Модель 2: En -> Ru ---
print("\n--- Обучение модели En -> Ru (PyTorch) ---")
inp_lang_en = targ_lang_en
targ_lang_ru = inp_lang_ru

input_tensor_en, target_tensor_ru, max_length_inp_en, max_length_targ_ru = load_dataset(
    path_to_file, num_examples, inp_lang_en, targ_lang_ru)
input_train_en, _, target_train_ru, _ = train_test_split(
    input_tensor_en, target_tensor_ru, test_size=0.2, random_state=42)

encoder_en_ru = Encoder(len(inp_lang_en.word2idx), embedding_dim, units).to(device)
decoder_en_ru = Decoder(len(targ_lang_ru.word2idx), embedding_dim, units).to(device)

checkpoint_dir_en_ru = './training_checkpoints_en_ru_pytorch'
train_model(input_train_en, target_train_ru, encoder_en_ru, decoder_en_ru, targ_lang_ru, EPOCHS, BATCH_SIZE, checkpoint_dir_en_ru)


# --- Итоговый конвейер ---
def translate_rus_eng_rus(sentence):
    print(f'\nИсходное предложение: {sentence}')
    intermediate_translation, _ = evaluate(sentence, encoder_ru_en, decoder_ru_en, inp_lang_ru, targ_lang_en, max_length_inp_ru, max_length_targ_en)
    print(f'Промежуточный перевод (En): {intermediate_translation}')
    final_translation, _ = evaluate(intermediate_translation, encoder_en_ru, decoder_en_ru, inp_lang_en, targ_lang_ru, max_length_inp_en, max_length_targ_ru)
    print(f'Итоговый перевод (Ru): {final_translation}')
    return final_translation

# Тестирование
translate_rus_eng_rus('я живу в липецке')
translate_rus_eng_rus('эта кошка умная')
translate_rus_eng_rus('я люблю машинное обучение') 