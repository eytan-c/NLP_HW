####Encoder
"""

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.bi_grus = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=False, bidirectional=True)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
             
        output, hidden = self.bi_grus(embedded, hidden)

        return output, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.n_layers*2, 1, self.hidden_size).cuda()
        return hidden

"""####Decoder with Attention"""

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        
        self.hidden_size = hidden_size 
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        attn_energies = torch.zeros(seq_len).cuda() # B x 1 x S
        
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies, 0).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):        
        energy = self.attn(encoder_output)        
        energy = hidden.view(-1).dot(energy.view(-1))
        return energy

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        # Keep parameters for reference
        self.hidden_size = hidden_size * 2      
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(input_size=embedding_size+self.hidden_size, hidden_size=self.hidden_size, num_layers=n_layers, batch_first=False)
        self.out = nn.Linear(self.hidden_size * 2, output_size)
        self.attn = Attn(self.hidden_size)
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        
        #rnn_output, hidden = self.lstm(rnn_input, last_hidden)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), 1)
        
        # Return final output, hidden state
        return output, context, hidden
      
    def init_hidden(self):
      hidden = torch.zeros(self.n_layers, 1, int(self.hidden_size)).cuda()
      return hidden
"""#### Save and Load models"""

def save_model(epoch):
  torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'forward_decoder_state_dict': forward_decoder.state_dict(),
            'backward_decoder_state_dict': backward_decoder.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            'forward_decoder_optimizer_state_dict': forward_decoder_optimizer.state_dict(),
            'backward_decoder_optimizer_state_dict': backward_decoder_optimizer.state_dict(),     
            'train_losses': train_losses,
            'val_losses': val_losses,
            'fk_belu_train': fk_belu_train,
            'fk_belu_val': fk_belu_val,
            'files_suffix': files_suffix,
            }, working_dir+"model_params_n_epochs_"+str(n_epochs)+".tar")
  print("Saved file:'"+model_params_file+" for epoch:",str(epoch))


def load_model():
  device = torch.device("cuda")
#   encoder = EncoderRNN(input_vocab.n_words, embedding_size, hidden_size, n_layers).cuda()
#   forward_decoder = AttnDecoderRNN(hidden_size, embedding_size, output_vocab.n_words, n_layers, dropout_p=dropout_p).cuda()
#   backward_decoder = AttnDecoderRNN(hidden_size, embedding_size, output_vocab.n_words, n_layers, dropout_p=dropout_p).cuda()
#   encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
#   forward_decoder_optimizer = optim.Adam(forward_decoder.parameters(), lr=learning_rate)
#   backward_decoder_optimizer = optim.Adam(backward_decoder.parameters(), lr=learning_rate)

  checkpoint = torch.load(working_dir+"model_params_n_epochs_"+str(n_epochs)+".tar")
  
  encoder.load_state_dict(checkpoint['encoder_state_dict'])
  forward_decoder.load_state_dict(checkpoint['forward_decoder_state_dict'])
  backward_decoder.load_state_dict(checkpoint['backward_decoder_state_dict'])
  
  encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
  forward_decoder_optimizer.load_state_dict(checkpoint['forward_decoder_optimizer_state_dict'])
  backward_decoder_optimizer.load_state_dict(checkpoint['backward_decoder_optimizer_state_dict'])
  
  epoch = checkpoint['epoch']
  train_losses = checkpoint['train_losses']
  val_losses = checkpoint['val_losses']
  fk_belu_train = checkpoint['fk_belu_train']
  fk_belu_val = checkpoint['fk_belu_val']
  files_suffix = checkpoint['files_suffix']
  
  encoder.to(device)
  forward_decoder.to(device)
  backward_decoder.to(device)
  encoder.train()
  forward_decoder.train()
  backward_decoder.train()
  
  return encoder,forward_decoder,backward_decoder,encoder_optimizer,forward_decoder_optimizer,backward_decoder_optimizer,epoch, train_losses, val_losses, fk_belu_train, fk_belu_val, files_suffix
