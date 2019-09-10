####Train function
"""

def train(input_tensor, backward_target_tensor, forward_target_tensor, simplified_word, forward_decoder, backward_decoder,
          encoder_optimizer, forward_decoder_optimizer, backward_decoder_optimizer, criterion, teacher_forcing_ratio=1, max_length=MAX_LENGTH):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    backward_decoder_optimizer.zero_grad()
    forward_decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word
    
    # Get size of input and target sentences
    input_length = input_tensor.size()[0]
    backward_target_length = backward_target_tensor.size()[0]
    forward_target_length = forward_target_tensor.size()[0]
    
    loss += run_model(input_tensor, backward_target_tensor, forward_target_tensor, input_length, backward_target_length, forward_target_length,
                      simplified_word, backward_decoder, forward_decoder, criterion)

    # Backpropagation
    loss.backward()
    encoder_optimizer.step()
    backward_decoder_optimizer.step()
    forward_decoder_optimizer.step()
    
    return loss.item() / (backward_target_length+forward_target_length)

def run_model(input_tensor, backward_target_tensor, forward_target_tensor, input_length, backward_target_length, forward_target_length, simplified_word, backward_decoder,
              forward_decoder, loss_func, teacher_forcing_ratio=teacher_forcing_ratio): 
  loss = 0 # Added onto for each word

  # Run words through encoder
  encoder_hidden = encoder.init_hidden()
  encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
  
  # Prepare input and output variables
  #forward_decoder_input = torch.LongTensor([[output_vocab.index_word(simplified_word, write=False)]]).cuda() #forward pass from y_s
  #forward_decoder_context = torch.zeros(1, forward_decoder.hidden_size).cuda()
  #backward_decoder_context = torch.zeros(1, forward_decoder.hidden_size).cuda()
  #backward_decoder_input = torch.LongTensor([[output_vocab.index_word(simplified_word, write=False)]]).cuda()
  forward_decoder_input = simplified_word.cuda() #forward pass from y_s
  forward_decoder_context = torch.zeros(1, forward_decoder.hidden_size).cuda()
  backward_decoder_input = simplified_word.cuda()
  backward_decoder_context = torch.zeros(1, forward_decoder.hidden_size).cuda()

  backward_decoder_hidden = encoder_hidden.view(-1,1,encoder_hidden.shape[2]*2) # Use last hidden state from encoder to start decoder
  #change here #forward_decoder_hidden = encoder_hidden.view(-1,1,encoder_hidden.shape[2]*2) # Use last hidden state from encoder to start decoder

  # Run model on input

  # Choose whether to use teacher forcing (Teacher forcing: Use the ground-truth target as the next input)
  use_teacher_forcing = random.random() < teacher_forcing_ratio

  # Backward pass
  for di in range(backward_target_length):
      backward_decoder_output, backward_decoder_context, backward_decoder_hidden = backward_decoder(
          backward_decoder_input, backward_decoder_context, backward_decoder_hidden, encoder_outputs)
      target_lables = torch.ones([1,300]).cuda()
      loss += loss_func(backward_decoder_output[0].view(-1).unsqueeze(0), backward_target_tensor[di].unsqueeze(0),target_lables)
      if use_teacher_forcing:
        backward_decoder_input = backward_target_tensor[di] # Next target is next input
      else:
        next_word = get_word_from_embeddings(backward_decoder_output.cuda())
        #backward_decoder_input = get_embeddings(next_word).cuda()
        backward_decoder_input = get_embeddings(next_word).cuda()

        # Stop at start of sentence (not necessary when using known targets)
        if next_word == '<SOS>': break

  # Forward pass
  forward_decoder_hidden = backward_decoder_hidden # change here
  for di in range(forward_target_length):
      forward_decoder_output, forward_decoder_context, forward_decoder_hidden = forward_decoder(
          forward_decoder_input, forward_decoder_context, forward_decoder_hidden, encoder_outputs)
      target_lables = torch.ones([1,300]).cuda()
      loss += loss_func(forward_decoder_output[0].view(-1).unsqueeze(0), forward_target_tensor[di].unsqueeze(0),target_lables)
      if use_teacher_forcing:
        forward_decoder_input = forward_target_tensor[di] # Next target is next input
      else:
        next_word = get_word_from_embeddings(forward_decoder_output.cuda())
        forward_decoder_input = get_embeddings(next_word).cuda()
        # Stop at start of sentence (not necessary when using known targets)
        if next_word == '<EOS>': break

    
  return loss

"""#### Evaluation functions"""

def evaluate_val_loss(backward_decoder, forward_decoder,teacher_forcing_ratio=0):
    backward_decoder.eval()
    forward_decoder.eval()
    encoder.eval()
    val_loss_sum = 0
    fk_belu = 0
    
    for i in range(len(X_val)):    
      pair = (X_val[i],y_val[i])         
      # Finding the complex word x_c and setting simple word y_s    
      simplified_word, org_index , target_index = get_simple_word(pair[0],pair[1]) 
#       input_tensor = torch.tensor([SOS_TOKEN]+(input_vocab.index_sentence(pair[0], write=False))+[EOS_TOKEN]).cuda()
#       backward_target_tensor = torch.tensor((output_vocab.index_sentence(pair[1], write=False)[target_index-1::-1])+[SOS_TOKEN]).cuda() # from y_s-1 to SOS
#       forward_target_tensor = torch.tensor((output_vocab.index_sentence(pair[1], write=False)[target_index+1:])+[EOS_TOKEN]).cuda() # from y_s+1 to EOS
      simplified_word = get_embeddings(simplified_word)
      input_tensor =  torch.cat((get_embeddings('<SOS>'), torch.cat((get_sentence_embeddings(pair[0]),get_embeddings('<EOS>'))))).cuda()
      backward_target_tensor = torch.cat((reverse_and_slice_tensor(get_sentence_embeddings(pair[1]), target_index), get_embeddings('<SOS>'))).cuda() # from y_s-1 to SOS
      forward_target_tensor = torch.cat((get_sentence_embeddings(pair[1])[target_index+1:],get_embeddings('<EOS>'))).cuda() # from y_s+1 to EOS

      # Get size of input and target sentences
      input_length = input_tensor.size()[0]
      backward_target_length = backward_target_tensor.size()[0]
      forward_target_length = forward_target_tensor.size()[0]
      
      cur_val_loss = run_model(input_tensor, backward_target_tensor, forward_target_tensor, input_length, backward_target_length, forward_target_length,
                               simplified_word, backward_decoder, forward_decoder, nn.CosineEmbeddingLoss())
      val_loss_sum += cur_val_loss.item() / (backward_target_length+forward_target_length) 
      
      #fk_belu += FKBLEU([pair[0].split(' ')], [pair[1].split(' ')], evaluate(pair[0])[1:-1])
    
    backward_decoder.train()
    forward_decoder.train()
    encoder.train()
    
    return val_loss_sum/len(X_val), fk_belu/len(X_val)

def run_decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs, max_length=MAX_LENGTH, is_backward=None):
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length).cuda()
    
    # Run through decoder
    for di in range(max_length):
        if is_backward:
          decoder_output, decoder_context, decoder_hidden = backward_decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        else:
          decoder_output, decoder_context, decoder_hidden = forward_decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        # Choose top word from output
        next_word = get_word_from_embeddings(decoder_output.cuda())
        decoded_words.append(next_word)
        
        if (next_word == '<SOS>' and is_backward) or (next_word == '<EOS>' and not is_backward):
            break
            
        # Next input is chosen word
        decoder_input = get_embeddings(next_word).cuda()
        
    return decoded_words

def evaluate(sentence, max_length=MAX_LENGTH):
      
    #return sentence.strip().split(' ')
    backward_decoder.eval()
    forward_decoder.eval()
    encoder.eval()
    
    #input_tensor = torch.tensor(input_vocab.index_sentence(sentence)).cuda()
    input_tensor =  torch.cat((get_embeddings('<SOS>'), torch.cat((get_sentence_embeddings(sentence),get_embeddings('<EOS>'))))).cuda()
    input_length = input_tensor.size()[0]
    
    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
    # Finding the complex word x_c and setting simple word y_s    
    simplified_word, org_index = get_simple_word(sentence)   
    simplified_word_embd = get_embeddings(simplified_word)
    
    # Create starting vectors for backward decoder
    backward_decoder_input = simplified_word_embd.cuda() # y_s
    backward_decoder_context = torch.zeros(1, backward_decoder.hidden_size).cuda()
    backward_decoder_hidden = encoder_hidden.view(-1,1,encoder_hidden.shape[2]*2) # Use last hidden state from encoder to start decoder
    
    # Create starting vectors for forward decoder
    forward_decoder_input = simplified_word_embd.cuda() # SOS
    forward_decoder_context = torch.zeros(1, forward_decoder.hidden_size).cuda()
    forward_decoder_hidden = encoder_hidden.view(-1,1,encoder_hidden.shape[2]*2) # Use last hidden state from encoder to start decoder
    
    # Run through decoders
    backward_words = run_decoder(backward_decoder_input, backward_decoder_context, backward_decoder_hidden, encoder_outputs, max_length, is_backward=True)
    forward_words = run_decoder(forward_decoder_input, forward_decoder_context, forward_decoder_hidden, encoder_outputs, max_length, is_backward=False)
    
    # Bulid sentance
    decoded_words = backward_words[::-1]+[simplified_word]+forward_words
    
    backward_decoder.train()
    forward_decoder.train()
    encoder.train()
    
    return decoded_words

def evaluate_randomly():
    #pair = random.choice(pairs)
    i = random.randint(0,len(X_val)-1)
    pair = [X_val[i],y_val[i]]
  
    output_words = evaluate(pair[0])
    print("pair",pair)
    print("output_words",output_words)
    output_sentence = ' '.join(output_words)
    
    print("Validation example")
    print('FKBLEU: ', FKBLEU([pair[0].split(' ')], [pair[1].split(' ')], output_words[1:-1]))
    if pickle_python2:
      print(pair[0].encode('utf-8'))
      print(pair[1].encode('utf-8'))
      print(output_sentence.encode('utf-8'))
    else:
      print(pair[0])
      print(pair[1])
      print(output_sentence)
    
def evaluate_randomly_train():
    #pair = random.choice(pairs)
    i = random.randint(0,len(X_val)-1)
    pair = [X_train[i],y_train[i]]
  
    output_words = evaluate(pair[0])
    output_sentence = ' '.join(output_words)
    
    print("Train example")
    print('FKBLEU: ', FKBLEU([pair[0].split(' ')], [pair[1].split(' ')], output_words[1:-1]))
    if pickle_python2:
      print(pair[0].encode('utf-8'))
      print(pair[1].encode('utf-8'))
      print(output_sentence.encode('utf-8'))
    else:
      print(pair[0])
      print(pair[1])
      print(output_sentence)

"""#### FK BELU"""

def iBLEU(input_sent, reference, candidate, alpha=0.9):
	"""
	Calculate iBLEU according to Xu et. al. 2016
	:param input_sent: original sentence
	:param reference: the target sentences to test by
	:param candidate: a proposed sentence from the input
	:param alpha: default param 0.9 from Sun and Zhou (2012)
	:return:
	"""
	smooth = bleu_score.SmoothingFunction()
	if len(candidate) < 2:
		return 0.0
	ref_candidate = bleu_score.sentence_bleu(reference, candidate, smoothing_function=smooth.method7)
	input_candidate = bleu_score.sentence_bleu(input_sent, candidate, smoothing_function=smooth.method7)
	# print('ref_candidate:', '%s; ' % ref_candidate, 'input_candidate:','%s; ' % input_candidate)
	# print('iBLEU (alpha * ref_candidate - (1 - alpha) * input_candidate) =\n %s' % (alpha * ref_candidate - (1 - alpha) * input_candidate))
	return alpha * ref_candidate - (1 - alpha) * input_candidate

	

def FK(text, language='heb'):
	"""
	TODO: count syllables not with heuristic
	Calculate Flesch-Kincaid Index (Kincaid et al 1975) according to Xu et. al. 2016
	:param language: Used for syllables count heuristic
	:param text: Assumes is a list of lists of words (list of sentences as lists of words)
	:return:
	"""
	# if isinstance(text, list) and all(isinstance(sen, list) for sen in text):
	# 	for sen in text:
	# 		assert all(isinstance(w, str) for w in sen)
	
	num_words = 0
	num_sents = 0
	num_syllables = 0
	if language == 'eng':
		parser = pyphen.Pyphen('en_us')
	else:
		parser = None
	# Gather numerical calculations
	for sen in text:
		num_sents += 1
		for word in sen:
			num_words += 1
			if language == 'heb':  # heuristic that each letter is a syllable in hebrew
				num_syllables += len(word)
			elif language == 'eng':  # syllable parser for English
				num_syllables += len(parser.inserted(word).split('-'))
	#print('Words, Sents, Syllables: %s, %s, %s' % (num_words, num_sents, num_syllables))
	return 0.39 * (num_words / num_sents) + 11.8 * (num_syllables / num_words) - 15.59


def FKdiff(input_sent, candidate):
  """
  
  :param input_sent: Assumes input sent is a list of lists of words
  :param candidate: Assumes candidate is a list of words
  :return:
  """
  # using torch
  # print('FKdiff:', torch.sigmoid(FK([candidate]) - FK(input_sent)))
  # return torch.nn.functional.sigmoid(FK([candidate]) - FK(input_sent))
  # using python native
  x = FK([candidate]) - FK(input_sent)
  # print('FK(candidate): %s;  FK(input_sent): %s' % (FK([candidate]), FK(input_sent)))
  # print('x: %s' % x)
  # print('FKdiff:', 1 / (1 + numpy.exp(-x)))
  return 1 / (1 + numpy.exp(-x))


def FKBLEU(input_sent, references, candidate):
	"""
	Calculate iBLEU according to Xu et. al. 2016
	:param input_sent: original sentence. Assumes list of lists of words
	:param references: the target sentences to test by. Assumes list of lists of words
	:param candidate: a proposed sentence from the input. List of words.
	:return:
	"""
	#print('Input: %s, refrences: %s, candidate: %s' % (input_sent, references, candidate))
	return iBLEU(input_sent, references, candidate) * FKdiff(input_sent, candidate)

"""#### Create results files"""

# Write results to file
def evaluate_validation():
  fk_belu = 0
  i_belu = 0
  fk = 0
  f = open(results_dir+'Results'+files_suffix+'.txt', 'w+')
  
  for i in range(len(X_val)):
    pair = [X_val[i],y_val[i]]
    output_words = evaluate(pair[0])
    output_sentence = ' '.join(output_words)
    f.write("This is line %d\r\n" % (i+1))
    
    if pickle_python2:
      f.write(pair[0].encode('utf-8')+" \n")
      f.write(pair[1].encode('utf-8')+" \n")
      f.write(output_sentence.encode('utf-8')+" \n")
    else:
      f.write(pair[0]+" \n")
      f.write(pair[1]+" \n")
      f.write(output_sentence+" \n")
    

    input_sent = [pair[0].split(' ')]
    references = [pair[1].split(' ')]
    candidate = output_words[1:-1]
    fk_belu += FKBLEU(input_sent, references , candidate)
    i_belu += iBLEU(input_sent, references, candidate)
    fk += FK(candidate)
  print("###########################################")
  print("\n This is FK-BELU on the validation %.18f \n" % (fk_belu/len(X_val)))
  print("\n This is iBELU on the validation %.18f \n" % (i_belu/len(X_val)))
  print("\n This is FK on the validation %.18f \n" % (fk/len(X_val)))
  f.write("\n This is FK-BELU on the validation %.18f \n" % (fk_belu/len(X_val)))
  f.write("\n This is i_belu on the validation %.18f \n" % (i_belu/len(X_val)))
  f.write("\n This is FK on the validation %.18f \n" % (fk/len(X_val)))
  f.write("\n Those are the run parameters: LR %.5f, dropout %.5f, n_layers %d, embedding_size %d,hidden_size %d \n" % (learning_rate,dropout_p,n_layers,embedding_size,hidden_size))
  print("Wrote those results to file")
  f.close()

class Plotter:
    def __init__(self, first_losses, fist_label, sec_losses, sec_label):
        self.first_losses = first_losses
        self.fist_label = fist_label
        self.sec_losses = sec_losses
        self.sec_label = sec_label

    def plot(self, file_name):
        """Plot the loss per epoch"""
        line1, = plt.plot(range(len(self.first_losses)), self.first_losses, label=self.fist_label)
        line2, = plt.plot(range(len(self.sec_losses)), self.sec_losses, label=self.sec_label)
        plt.legend(handles=[line1,line2])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Cosine Embedding Loss')
        plt.grid(True)
        plt.savefig(file_name)
