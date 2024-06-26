#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
from __future__ import unicode_literals, print_function, division
import time
import random
import wandb

import torch
import torch.nn as nn
from torch import optim
import os
#os.chdir(os.getcwd()+'/translation')

import data
import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = data.tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == data.EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def eval_epoch(dataloader, encoder, decoder, criterion):

    total_loss = 0
    with torch.no_grad():
       for data in dataloader:
            input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, eval_dataloader, encoder, decoder, n_epochs, learning_rate=0.0001,
               print_every=5):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss_t = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        loss_e = eval_epoch(eval_dataloader, encoder, decoder, criterion)
        wandb.log({"train":{"loss":loss_t},"eval":{"loss":loss_e}})
        print_loss_total += loss_t
        plot_loss_total += loss_t

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (data.timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

    
if __name__ == "__main__":
    
    config = {'epochs':80,
              'batch_size':320,
              'hidden_size':128, 
              'learning_rate':0.0001,
              'max_length':10,
              'len_dataset':150000,
              'split_val':0.2}
    
    wandb.login()
    wandb.init (project='translation', name='experiment4', config = config )    

    input_lang, output_lang, train_dataloader, eval_dataloader = data.get_dataloader(wandb.config)

    encoder = model.EncoderRNN(input_lang.n_words, wandb.config.hidden_size).to(device)
    decoder = model.AttnDecoderRNN(wandb.config.hidden_size, output_lang.n_words).to(device)

    train(train_dataloader, eval_dataloader, encoder, decoder, n_epochs=wandb.config.epochs, learning_rate=wandb.config.learning_rate)    