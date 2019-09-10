####Models' init
"""

# Data Load

try:
  if not load_from_pickle:
    raise IOError("Do not load from pickle!")
  if train_LM:
    if pickle_python2:
      embeddings, input_vocab_n_words, output_vocab_n_words, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test, X_lm, y_lm = pickle.load(open(working_dir+"pre_trained_data_with_LM_python2.pickle", "rb"))
    else:
      embeddings, input_vocab, output_vocab, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test, X_lm, y_lm = pickle.load(open(working_dir+"pre_trained_data_with_LM.pickle", "rb"))
  else:
    if pickle_python2:
      embeddings, input_vocab_n_words, output_vocab_n_words, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(open(working_dir+"pre_trained_data_python2.pickle", "rb"))
    else:
      embeddings, input_vocab, output_vocab, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(open(working_dir+"pre_trained_data.pickle", "rb"))
except (OSError, IOError) as e:
  input_vocab = Vocab("Heb-reg")
  output_vocab = Vocab("Heb-simple")
  dic_reg_to_sim = {}
  if train_LM:
    embeddings, input_vocab, output_vocab, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test, X_lm, y_lm = init_data(input_vocab,output_vocab,dic_reg_to_sim)
    if pickle_python2:
      pickle.dump([embeddings, input_vocab.n_words, output_vocab.n_words, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test, X_lm, y_lm], open(working_dir+"pre_trained_data_with_LM_python2.pickle", "wb"), protocol=2)
    else:
      pickle.dump([embeddings, input_vocab, output_vocab, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test, X_lm, y_lm], open(working_dir+"pre_trained_data_with_LM.pickle", "wb"))
    
  else:
    embeddings, input_vocab, output_vocab, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test = init_data(input_vocab,output_vocab,dic_reg_to_sim)
    if pickle_python2:
      pickle.dump([embeddings, input_vocab.n_words, output_vocab.n_words, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test], open(working_dir+"pre_trained_data_python2.pickle", "wb"), protocol=2)
    else:
      pickle.dump([embeddings, input_vocab, output_vocab, dic_reg_to_sim, X_train, y_train, X_val, y_val, X_test, y_test], open(working_dir+"pre_trained_data.pickle", "wb"))   
    
  print("Pickled all data")

  
# Initialize models
if pickle_python2:
  encoder = EncoderRNN(input_vocab_n_words, embedding_size, hidden_size, n_layers).cuda()
else:
  encoder = EncoderRNN(input_vocab.n_words, embedding_size, hidden_size, n_layers).cuda()
# forward_decoder = AttnDecoderRNN(hidden_size, embedding_size, output_vocab.n_words, n_layers, dropout_p=dropout_p).cuda()
# backward_decoder = AttnDecoderRNN(hidden_size, embedding_size, output_vocab.n_words, n_layers, dropout_p=dropout_p).cuda()
forward_decoder = AttnDecoderRNN(hidden_size, embedding_size, embedding_size, n_layers, dropout_p=dropout_p).cuda()
backward_decoder = AttnDecoderRNN(hidden_size, embedding_size, embedding_size, n_layers, dropout_p=dropout_p).cuda()

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
forward_decoder_optimizer = optim.Adam(forward_decoder.parameters(), lr=learning_rate)
backward_decoder_optimizer = optim.Adam(backward_decoder.parameters(), lr=learning_rate)
# criterion = nn.NLLLoss()
criterion = nn.CosineEmbeddingLoss()

"""####Training as LM"""

if train_LM:
  print("Training model as LM...")
  
  if load_model_flag:
    print("Loading model's parameters in LM...")
    encoder,forward_decoder,backward_decoder,encoder_optimizer,forward_decoder_optimizer,backward_decoder_optimizer,epoch, train_losses, val_losses, files_suffix = load_model()
    print("Training LM from epoch: ",epoch)
  else:
    epoch = 1
  
  f = open(results_dir+'Train_LM_loss_'+files_suffix+'.txt', 'w+')
  
  while epoch < train_LM_n_epochs + 1:
      for i in range(len(X_lm)):
        pair = (X_lm[i],y_lm[i])        
        splitting_index = (len(pair[0].split(' '))-1)//2 # mid sentance so both decoders would learn to generate words
        
        #simplified_word = output_vocab.index_word(pair[1].split(' ')[splitting_index])
        simplified_word = get_embeddings(pair[1].split(' ')[splitting_index])

#         input_tensor = torch.tensor([SOS_TOKEN]+(input_vocab.index_sentence(pair[0], write=False))+[EOS_TOKEN]).cuda()
#         backward_target_tensor = torch.tensor((output_vocab.index_sentence(pair[1], write=False)[splitting_index-1::-1])+[SOS_TOKEN]).cuda() # from y_s-1 to SOS
#         forward_target_tensor = torch.tensor((output_vocab.index_sentence(pair[1], write=False)[splitting_index+1:])+[EOS_TOKEN]).cuda() # from y_s+1 to EOS
        
        input_tensor =  torch.cat((get_embeddings('<SOS>'), torch.cat((get_sentence_embeddings(pair[0]),get_embeddings('<EOS>'))))).cuda()
        backward_target_tensor = torch.cat((reverse_and_slice_tensor(get_sentence_embeddings(pair[1]), splitting_index), get_embeddings('<SOS>'))).cuda() # from y_s-1 to SOS
        forward_target_tensor = torch.cat((get_sentence_embeddings(pair[1])[splitting_index+1:],get_embeddings('<EOS>'))).cuda() # from y_s+1 to EOS

        # Run the train function
        loss = train(input_tensor, backward_target_tensor, forward_target_tensor, simplified_word, forward_decoder, backward_decoder,
                     encoder_optimizer, forward_decoder_optimizer, backward_decoder_optimizer, criterion,train_LM_teacher_forcing_ratio)

        # Keep track of loss
        print_loss_total += loss
        if i == 0: continue
          

        if i % 100 == 0:
            print_loss_avg = print_loss_total / i
            print('(Epoch %d)(pair number %d) %.4f' % (epoch,i, print_loss_avg))

      if epoch == 0: continue

      if epoch % print_every == 0:
          print_loss_avg = print_loss_total / (print_every * len(X_lm))
          print_loss_total = 0
          print('(Epoch %d) %.4f' % (epoch, print_loss_avg))
          f.write("(Epoch %d) %.4f \n" % (epoch, print_loss_avg))
      epoch += 1
      save_model(epoch)

  print("Done!")
  f.close()

"""#### Training"""

# Begin!

print("Training...")

if load_model_flag and not train_LM:
  print("Loading model's parameters...")
  encoder,forward_decoder,backward_decoder,encoder_optimizer,forward_decoder_optimizer,backward_decoder_optimizer,epoch, train_losses, val_losses, files_suffix = load_model()
  print("Training from epoch: ",epoch)
else:
  epoch = 1


f = open(results_dir+'Train_loss_'+files_suffix+'.txt', 'w+')
f_2 = open(results_dir+'Validation_loss_'+files_suffix+'.txt', 'w+')

while epoch < n_epochs + 1:
    
    # Get training data for this cycle
    for i in range(len(X_train)):
      pair = (X_train[i],y_train[i])
      
      # Finding the complex word x_c and setting simple word y_s    
      simplified_word, org_index , target_index = get_simple_word(pair[0],pair[1]) 

#       input_tensor = torch.tensor([SOS_TOKEN]+(input_vocab.index_sentence(pair[0], write=False))+[EOS_TOKEN]).cuda() # with EOS in the end
#       backward_target_tensor = torch.tensor((output_vocab.index_sentence(pair[1], write=False)[target_index-1::-1])+[SOS_TOKEN]).cuda() # from y_s-1 to SOS
#       forward_target_tensor = torch.tensor((output_vocab.index_sentence(pair[1], write=False)[target_index+1:])+[EOS_TOKEN]).cuda() # from y_s+1 to EOS

      simplified_word = get_embeddings(simplified_word)
      input_tensor =  torch.cat((get_embeddings('<SOS>'), torch.cat((get_sentence_embeddings(pair[0]),get_embeddings('<EOS>'))))).cuda()
      backward_target_tensor = torch.cat((reverse_and_slice_tensor(get_sentence_embeddings(pair[1]), target_index), get_embeddings('<SOS>'))).cuda() # from y_s-1 to SOS
      forward_target_tensor = torch.cat((get_sentence_embeddings(pair[1])[target_index+1:],get_embeddings('<EOS>'))).cuda() # from y_s+1 to EOS

      # Run the train function
      loss = train(input_tensor, backward_target_tensor, forward_target_tensor, simplified_word, forward_decoder, backward_decoder,
                   encoder_optimizer, forward_decoder_optimizer, backward_decoder_optimizer, criterion,teacher_forcing_ratio)
            
      
      # Keep track of loss
      print_loss_total += loss
      
      if i == 0: continue

      if i % 100 == 0:
          print_loss_avg = print_loss_total / i
          print('(Epoch %d)(pair number %d) %.4f' % (epoch,i, print_loss_avg))

    if epoch == 0: continue

    if epoch % print_every == 0:
        # Estimate FKBELU
        fk_belu = 0
#         idx = random.sample(range(len(X_train)), min(1000,len(X_train))) # get 1000 random indexes
#         for i in idx: 
#           pair = (X_train[i],y_train[i])
#           fk_belu += FKBLEU([pair[0].split(' ')], [pair[1].split(' ')], evaluate(pair[0])[1:-1])
#         fk_belu /= min(1000,len(X_train))
        
        # Train loss
        print_loss_avg = print_loss_total / (print_every*len(X_train))
        print_loss_total = 0
        print('(Epoch %d) Train loss %.4f \n' % (epoch, print_loss_avg))
        f.write("(Epoch %d) Train loss %.4f \n" % (epoch, print_loss_avg))    
        train_losses.append(print_loss_avg) 
        fk_belu_train.append(fk_belu)
        
        # Validation loss
        val_loss, fk_belu = evaluate_val_loss(backward_decoder, forward_decoder,teacher_forcing_ratio=0)
        print('(Epoch %d) Validation loss %.4f \n' % (epoch, val_loss))
        f_2.write("(Epoch %d) %.4f \n" % (epoch, val_loss))
        val_losses.append(val_loss)
        fk_belu_val.append(fk_belu)
    epoch += 1
    save_model(epoch)
        
print("Done!")

f.close()
f_2.close()

"""#### Evaluate"""

for i in range(5):
    evaluate_randomly()
    print('\n')

for i in range(5):
    evaluate_randomly_train()
    print('\n')

# Write all evaluation results on validation data to file
evaluate_validation()

"""#### Plot"""

# Plotting Losses
file_name = results_dir+'Losses'+files_suffix+'.png'
plotter = Plotter(train_losses,"Train", val_losses, "Validation")
plotter.plot(file_name)
