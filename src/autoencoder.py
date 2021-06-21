import pandas as pd
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SEQLEN = 25
EMBEDDING_SIZE = 25

def generate_data():
    data = pd.read_excel('../dataset/minioutput.xlsx')
    data = data.sample(frac = 1, random_state=7).reset_index(drop=True)
    data.drop('Unnamed: 0', axis=1, inplace=True)
    final = data.copy()
    data.drop('Original', axis=1, inplace=True)
    data.drop('Cluster', axis=1, inplace=True)

    return data, final

def generate_input(data):
    text_input = data.pop('Text')

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='',
        lower=True, split=' ')
    tokenizer.fit_on_texts(text_input)

    x = tokenizer.texts_to_sequences(text_input)   
    res = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=25,
                                                        padding='post')
    vocab_size = len(tokenizer.word_index)

    standard_scaler = StandardScaler()
    data_standard = standard_scaler.fit_transform(data)

    final_data = {'data_input': data_standard, 'text_input': res}
    final_data_out = {'decoded_data': data_standard, 'decoded_txt': res}

    return final_data, final_data_out, vocab_size

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plot_name = '../results/' + metric + '.svg'
    #plt.savefig(plot_name)
    plt.show()

def autoencoder(final_data, final_data_out, vocab_size):
    len_data = final_data["data_input"].shape[1]

    data_input = tf.keras.layers.Input(shape=(len_data, ), name='data_input')

    text_input = tf.keras.layers.Input(shape=(SEQLEN,), name='text_input')
    x = tf.keras.layers.Embedding(vocab_size + 1, EMBEDDING_SIZE,
                                input_length=SEQLEN)(text_input)
    text_output = tf.keras.layers.LSTM(SEQLEN, activation='relu')(x)

    concat_inputs = tf.keras.layers.concatenate([data_input, text_output])
    encoded = tf.keras.layers.Dense(16, activation='relu')(concat_inputs)
    # encoded = tf.keras.layers.Dropout(0.2)(encoded)
    encoded = tf.keras.layers.Dense(8, activation='relu')(encoded)
    # encoded = tf.keras.layers.Dropout(0.2)(encoded)
    encoded = tf.keras.layers.Dense(4, activation='relu')(encoded)

    decoded = tf.keras.layers.Dense(4, activation='relu')(encoded)
    # decoded = tf.keras.layers.Dropout(0.2)(decoded)
    decoded = tf.keras.layers.Dense(8, activation='relu')(decoded)
    # decoded = tf.keras.layers.Dropout(0.2)(decoded)
    decoded = tf.keras.layers.Dense(16, activation='relu')(decoded)

    decoded_data = tf.keras.layers.Dense(len_data, name='decoded_data')(decoded)
    decoded_text = tf.keras.layers.Dense(SEQLEN, name='decoded_txt')(decoded)

    decoded = [decoded_data, decoded_text]

    ae_input_layers = {'data_input': data_input,
                    'text_input': text_input}
    ae_output_layers = {'decoded_data': decoded[0],
                        'decoded_txt': decoded[1]}

    autoencoder = tf.keras.Model(ae_input_layers, ae_output_layers)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                        loss='mse',
                        metrics=['mse', 'mae'])
    print(autoencoder.summary())

    history = autoencoder.fit(final_data, final_data_out,
                          epochs=100,
                          validation_split=0.2
                            )

    predicted = autoencoder.predict(final_data)

    metrics = ['loss', 'decoded_data_loss', 'decoded_txt_loss', 'decoded_data_mse', 'decoded_data_mae', 'decoded_txt_mse', 'decoded_txt_mae']
    for metric in metrics:
        plot_graphs(history, metric)
    
    return predicted

def outliers(final, predicted):
    final['decoded_txt'] = predicted['decoded_txt'].std(axis=1).tolist()
    final['raw'] = predicted['decoded_txt'].tolist()
    final['decoded_data'] = predicted['decoded_data'].mean(axis=1).tolist()
    final = final.sort_values(by=['decoded_txt', 'decoded_data']).reset_index(drop=True)

def main():
    data, final = generate_data()

    final_data, final_data_out, vocab_size = generate_input(data)
    
    predicted = autoencoder(final_data, final_data_out, vocab_size)

    outliers(final, predicted)

if __name__ == '__main__':
    main()