import os
import torch
import peft
import numpy as np
from utils.ext import update_causal_mask
from utils.partial_models import add_partial_forward_gpt2, add_partial_forward_bert, add_partial_forward_llama, \
    add_partial_forward_donut_decoder
from constants import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from utils.functional import get_layer_decomp

class ModelWrapper():
    def __init__(self, args):
        assert (args.model_path in ['naver-clova-ix/donut-base-finetuned-docvqa','bert-base-uncased', 'gpt2', 'openai-community/gpt2-large', 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']),\
            'Model is not yet supported - add it to assertion list and specify implementation details'
        # access_token = os.environ['HF_TOKEN']
        self.args = args
        model_kwargs = {'cache_dir': args.cache_dir} if args.cache_dir is not None else {}

        model_kwargs['pretrained_model_name_or_path'] = args.model_path if args.finetuned_path is None or args.train_method == 'lora' else args.finetuned_path
        model_kwargs['attn_implementation'] = args.attn_implementation

        if args.hidden_act is not None and args.model_path in ['gpt2', 'openai-community/gpt2-large']:
            model_kwargs['activation_function'] = args.hidden_act
        elif args.hidden_act is not None and args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']:
            model_kwargs['hidden_act'] = args.hidden_act
            
        if args.precision == '8bit':
            model_kwargs['load_in_8bit'] = True
        if args.precision == 'half':
            model_kwargs['torch_dtype'] = torch.float16
        if args.precision == 'double':
            model_kwargs['torch_dtype'] = torch.float64
        if args.task == 'seq_class':
            self.model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs).to(self.args.device)
        elif args.task == 'next_token_pred':
            if args.model_path in ['naver-clova-ix/donut-base-finetuned-docvqa']:
                from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig

                config_donut = VisionEncoderDecoderConfig.from_pretrained(args.model_path)
                config_donut.encoder.image_size = [1280, 960]
                config_donut.decoder.max_length = 128
                self.processor = DonutProcessor.from_pretrained(args.model_path)
                self.processor.feature_extractor.size = [960, 1280]
                self.processor.feature_extractor.do_align_long_axis = False
                self.model = VisionEncoderDecoderModel.from_pretrained(args.model_path, config=config_donut).to(self.args.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs).to(self.args.device)
        else:
            assert False
        g_cpu = torch.Generator(device=self.model.device)
        g_cpu.manual_seed(0)
        self.model.eval()
        if args.model_path in ['naver-clova-ix/donut-base-finetuned-docvqa']:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, cache_dir = args.cache_dir)
        # self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, token=access_token, cache_dir = args.cache_dir)
        self.tokenizer.model_max_length = 512
        
        if args.pad == 'left':
            self.tokenizer.padding_side = "left"
            
        if args.model_path in ['gpt2', 'openai-community/gpt2-large']:        
            self.start_token = None
            self.eos_token = self.model.config.eos_token_id
            self.layer_ids = list(range(4, 137, 12))
            
            if args.task == 'seq_class':
                self.model.score.weight.data.normal_( mean=0.0, std=1e-3, generator=g_cpu )
            
            # Set padding token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.pad_token = self.model.config.eos_token_id
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
            self.embeddings_weight_nopos = self.model.transformer.wte.weight.unsqueeze(0)
            
            self.emb_size = self.model.config.n_embd
            add_partial_forward_gpt2(self.model.transformer)
        elif args.model_path in ['naver-clova-ix/donut-base-finetuned-docvqa']:
            self.start_token = self.tokenizer.convert_tokens_to_ids("<s_docvqa>")
            self.eos_token = self.tokenizer.eos_token_id
            self.pad_token = self.tokenizer.pad_token_id

            self.model.config.pad_token_id = self.pad_token
            self.model.config.decoder_start_token_id = self.start_token

            self.layer_ids =  list(range(359,462,26))
            # self.layer_ids = list(range(369,462,26))
            # self.layer_ids = [359,369,385,395,411,421,437,447]

            donut_token_embeddings = self.model.decoder.model.decoder.embed_tokens.weight
            donut_position_embeddings = self.model.decoder.model.decoder.embed_positions.weight

            pos_first_row = donut_position_embeddings[0].unsqueeze(0)  # (1, 1024)
            embeddings_no_pos = donut_token_embeddings + pos_first_row  # broadcast

            # Suppose you want shape [1, vocab_size, 1, 1024]:
            self.embeddings_weight_nopos = embeddings_no_pos[None, :, None, :]

            self.emb_size = self.model.config.decoder.d_model  # Not applicable; not based on token input size
            # add_partial
            add_partial_forward_donut_decoder(self.model)
        elif args.model_path in ['bert-base-uncased']:
          
            self.start_token = 101
            self.eos_token = 102
            self.pad_token = 0
            self.layer_ids = list(range(5,190,16))
            
            # Store embeddings
            bert_embeddings_weight = self.model.bert.embeddings.word_embeddings.weight.unsqueeze(0)
            bert_embeddings_weight_token = self.model.bert.embeddings.token_type_embeddings.weight.unsqueeze(0)
            
            self.embeddings_weight_nopos = (bert_embeddings_weight_token + bert_embeddings_weight[0][:,None,:])[None,:,:,:]
            self.emb_size = self.model.config.hidden_size
            add_partial_forward_bert(self.model.bert)
        elif args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']:
            
            self.start_token = self.tokenizer.bos_token_id
            self.eos_token = self.tokenizer.eos_token_id
            if args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf']:
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.unk_token})
                self.pad_token = self.tokenizer.unk_token_id
                self.model.config.pad_token_id = self.tokenizer.unk_token_id
            else:
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
                self.pad_token = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            if args.train_method == 'lora' and args.finetuned_path is not None:
                lora_cfg = peft.LoraConfig(r=args.lora_r,target_modules=['q_proj'])
                self.model = peft.LoraModel(self.model, lora_cfg, 'default')
                self.model.load_state_dict(torch.load(args.finetuned_path, map_location=torch.device('cpu')))
                self.model = self.model.model
                self.layer_ids = list(range(0,64,2))
            else:
                if args.task == 'seq_class':
                    self.model.score.weight.data.normal_(mean=0.0, std=1e-3)
                #else:
                    #self.model.lm_head.weight.data.normal_(mean=0.0, std=1e-6)
                    
                if args.train_method == 'lora':
                    lora_cfg = peft.LoraConfig(r=args.lora_r,target_modules=['q_proj'])
                    self.full_model = peft.LoraModel(self.model, lora_cfg, "default")
                    self.model = self.full_model.model
                    self.layer_ids = list(range(1,64,2))
                    
                else:
                    self.layer_ids = list(range(1,281,9))
            
            self.emb_size = self.model.config.hidden_size
            self.embeddings_weight_nopos = self.model.model.embed_tokens.weight.unsqueeze(0)
            add_partial_forward_llama(self.model.model)
        
        self.trainable_parameters = lambda: (param for param in self.model.parameters() if param.requires_grad)
        config['START_TOKEN'] = self.start_token
        config['EOS_TOKEN'] = self.eos_token
        config['PAD_TOKEN'] = self.pad_token
        self.set_model_device(args.device)
        
    def compute_grads_fed_avg(self, batch, labels, create_graph=False):
        og_weights = [param.data.clone() for param in self.model.parameters()]

        self.model.eval()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.avg_lr)

        n_minib = batch['input_ids'].shape[0] // self.args.b_mini
        print(n_minib)
        for _ in range(self.args.avg_epochs):
            for i in range(n_minib):
                print(batch['input_ids'].shape)
                b_mini = {k:batch[k][i*self.args.b_mini:(i+1)*self.args.b_mini] for k in batch.keys()}
                y_mini = labels[:, i*self.args.b_mini:(i+1)*self.args.b_mini]
                print(b_mini['input_ids'].shape, y_mini)
                optimizer.zero_grad()
                outs = self.model(**b_mini, labels=y_mini)
                outs.loss.backward()
                optimizer.step()
           
        grad = [-(param.data.detach() - og_weights[i])/n_minib/self.args.avg_lr/self.args.avg_epochs for i, param in enumerate(self.model.parameters())]
        for i, param in enumerate(self.model.parameters()):
            param.data = og_weights[i]
        self.model.eval()
        return grad

    def move_to_device(self,batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, (tuple, list)):
            return type(batch)(self.move_to_device(x, device) for x in batch)
        elif isinstance(batch, dict):
            return {k: self.move_to_device(v, device) for k, v in batch.items()}
        else:
            return batch  # leave untouched if not tensor-like

    def compute_grads(self, batch, y_labels, create_graph=False):
        outs = []
        if self.args.grad_mode == 'eval':
            self.model.eval()
        else:
            self.model.train()
        dev = y_labels.device
        if self.args.precision != '8bit':
            batch = self.move_to_device(batch,self.args.device_grad)
            y_labels = y_labels.to(self.args.device_grad)
            self.model.to(self.args.device_grad)
        if self.args.task == 'next_token_pred':
            if self.args.dataset == 'donut':
                labels = y_labels
                # Store pixel values in the wrapper
                self.set_last_pixel_values(batch['pixel_values'])
            else:
                labels = torch.where(batch['attention_mask'].bool(), batch['input_ids'], -100)
        elif self.args.task == 'seq_class':
            labels = y_labels
        if self.args.grad_b is None:
            if self.args.algo == 'fedavg':
                grad=self.compute_grads_fed_avg(batch, labels, create_graph)
            elif self.is_lower():
                outputs = self.model(**batch)
                logits = outputs.logits.float()
                loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits, dim=-1), labels.squeeze(0))
                for name, param in self.model.named_parameters():
                    param.requires_grad=True
                    print(name, param.shape, param.requires_grad)
                grad = torch.autograd.grad(loss, self.trainable_parameters(), create_graph=create_graph, allow_unused=True)
                
            else:
                self.model = self.model.to(self.args.device)
                batch = self.move_to_device(batch,self.args.device)
                labels = labels.to(self.args.device)
                outs = self.model(**batch, labels=labels, output_hidden_states=True)
                    
                if self.args.loss == 'mse':
                    loss = outs.hidden_states[-1].pow(2).mean()
                elif self.args.loss == 'ce':
                    loss = outs.loss
                grad = torch.autograd.grad(loss, self.trainable_parameters(), create_graph=create_graph, allow_unused=True)

        else:
            
            minib_size = self.args.batch_size // self.args.grad_b
            for i in range(self.args.grad_b):
                mini_batch = {k: batch[k][i*minib_size:(i+1)*minib_size] for k in batch.keys()}
                if self.is_lower():
                    outputs = self.model(**mini_batch)
                    logits = outputs.logits.float()
                    loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(logits, dim=-1), labels.squeeze(0)[i*minib_size:(i+1)*minib_size])
                    loss.backward()
                else:
                    outs = self.model(**mini_batch, labels=labels[:,i*minib_size:(i+1)*minib_size])
                    outs.loss.backward()
            grad = tuple([param.grad.detach().cpu()/self.args.grad_b for param in self.model.parameters()])
        self.set_model_device(dev)
        if self.args.precision != '8bit':
            batch = self.move_to_device(batch,self.args.device_grad)
            y_labels = y_labels.to(dev)
        self.model.eval()
        #torch.save(grad, f'./grad_{self.args.algo}1.pt')
        #raise ValueError
        if self.args.dataset == 'donut':
            return grad, outs.encoder_hidden_states[-1]
        else:
            return grad
            
    def set_model_device(self, device):
        if self.args.precision == '8bit':
            return
        if self.args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B'] and device!='cpu':
            self.model.model.embed_tokens.to(device)
            self.model.model.rotary_emb.to(device)
            for i in range(self.args.n_layers):
                self.model.model.layers[i].to(device)
        else:
            self.model.to(device)

    def get_matrices_expansions(self, true_grads, B=None, tol=None):
        if B is None:
            max_rank = 0
            for i in self.layer_ids[:10]:
                grad = true_grads[i]
                if self.args.model_path in ['gpt2', 'openai-community/gpt2-large']:
                    grad = grad.T
                if self.args.precision == 'half':
                    B = np.linalg.matrix_rank( grad.float().cpu() , tol=tol)
                else:
                    B = np.linalg.matrix_rank( grad.cpu() , tol=tol)
                if max_rank < B:
                    max_rank = B
            B = max_rank
        if self.args.algo == 'fedavg':
            B += 60
        B = min(B, self.emb_size - self.args.rank_cutoff)
        
        R_Qs = []
        
        for i in range(self.args.n_layers):
            grad_Q = true_grads[self.layer_ids[i]]
            if self.args.model_path in ['gpt2', 'openai-community/gpt2-large']: # ,"naver-clova-ix/donut-base-finetuned-docvqa"]:
                grad_Q = grad_Q.T
            _, R_Q = get_layer_decomp(grad_Q, B=B, tol=tol, upcast=(self.args.precision=='half'))
            R_Q = R_Q.to(self.args.device)
            R_Qs.append(R_Q)
        return B, R_Qs
            
    def get_embeddings(self, pos = None):
        self.model = self.model.to(self.args.device)
        if self.args.model_path in ['bert-base-uncased']:
            bert_embeddings_weight_position = self.model.bert.embeddings.position_embeddings.weight.unsqueeze(0)
            emb = self.embeddings_weight_nopos.to(self.args.device) + bert_embeddings_weight_position[0][pos:pos+1,None,None,:]
            emb = self.model.bert.embeddings.LayerNorm(emb)
            return emb
        
        elif self.args.model_path in ['gpt2', 'openai-community/gpt2-large']:
            gpt_embeddings_weight_position = self.model.transformer.wpe.weight.unsqueeze(0)
            emb = self.embeddings_weight_nopos.to(self.args.device) + gpt_embeddings_weight_position[0][pos:pos+1,None,:]
            emb = self.model.transformer.h[0].ln_1(emb)
            return emb
        elif self.args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']:
            emb = self.embeddings_weight_nopos.to(self.args.device)
            return self.model.model.layers[0].input_layernorm(emb)
        elif self.args.model_path in ['naver-clova-ix/donut-base-finetuned-docvqa']:
            # Get token embeddings and positional embeddings from the decoder.
            donut_token_embeddings = self.model.decoder.model.decoder.embed_tokens.weight  # shape: (vocab_size, hidden_dim)
            donut_pos_embeddings = self.model.decoder.model.decoder.embed_positions.weight  # shape: (max_positions, hidden_dim)

            # Assume that self.embeddings_weight_nopos is a tensor of shape (vocab_size, hidden_dim)
            # Here we add the positional embedding for the given position.
            # [pos:pos+1] extracts a slice with shape (1, hidden_dim) so that broadcasting works correctly.
            emb = donut_token_embeddings.to(self.args.device) + donut_pos_embeddings[pos:pos + 1].to(self.args.device)

            # If the decoder applies a layer norm on its embeddings (often named layernorm_embedding),
            # you can mimic that by applying it here.
            if hasattr(self.model.decoder.model.decoder, 'layernorm_embedding'):
                emb = self.model.decoder.model.decoder.layernorm_embedding(emb.to(self.args.device))
            # Otherwise, you might also try the alternative layer norm (e.g., layer_norm)
            elif hasattr(self.model.decoder.model.decoder, 'layer_norm'):
                emb = self.model.decoder.model.decoder.layer_norm(emb.to(self.args.device))

            return emb.unsqueeze(0)



    def get_layer_inputs(self, sentences, token_type_ids=None, attention_mask=None, layers=1, encoder_last_hidden_state=None):
        if self.args.model_path in ['bert-base-uncased']:
            # if token_type_ids is None:
            #     raise ValueError('Token type must be defined when model is BERT')
            # emb = self.model.bert.embeddings( input_ids=sentences, token_type_ids=token_type_ids )
            # layer_inputs = []
            # for i in range(layers):
            #     emb = self.model.bert.encoder.layer[i](emb)[0]# As end of sentence tokens have little gradient they are unreliable measures for sentence inclusion
            #     layer_inputs.append(emb[ : , :-1, : ].clone())
            # return layer_inputs
            return self.model.bert.get_hidden_states(input_ids=sentences, token_type_ids=token_type_ids, n_layers=layers)
        
        elif self.args.model_path in ['gpt2', 'openai-community/gpt2-large']:
            return self.model.transformer.get_hidden_states(input_ids=sentences, attention_mask=attention_mask, n_layers=layers)
        
        elif self.args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']:
            position_ids = torch.arange(sentences.size(1)).unsqueeze(0).repeat(sentences.size(0), 1).to(self.args.device)
            # if attention_mask is not None:
            #     first_item_idx = torch.argmax(attention_mask, dim=1).unsqueeze(1)
            #     position_ids = torch.maximum(position_ids - first_item_idx, torch.tensor(0).to(self.args.device))
            #     attention_mask = update_causal_mask(self.model.model, attention_mask, emb).to(self.args.device)

            # layer_inputs = []
            # for i in range(layers):
            #     emb = self.model.model.layers[i](emb, attention_mask=attention_mask, position_ids=position_ids)[0]# As end of sentence tokens have little gradient they are unreliable measures for sentence inclusion
            #     layer_inputs.append(self.model.model.layers[i+1].input_layernorm(emb))
            # return layer_inputs
            return self.model.model.get_hidden_states(input_ids=sentences, position_ids=position_ids,attention_mask=attention_mask, n_layers=layers)
        elif self.args.model_path in ['naver-clova-ix/donut-base-finetuned-docvqa']:
            # self.model.decoder.get_hidden_states(input_ids=sentences, encoder_hidden_states=encoder_last_hidden_state, attention_mask=attention_mask, n_layers=layers)
            encoder_last_hidden_state = encoder_last_hidden_state.to(self.args.device) if encoder_last_hidden_state is not None else None
            sentences = sentences.to(self.args.device)
            attention_mask = attention_mask.to(self.args.device) if attention_mask is not None else None
            # return self.model.decoder(input_ids=sentences, encoder_hidden_states=encoder_last_hidden_state, attention_mask=attention_mask).hidden_states
            return self.model.decoder(input_ids=sentences, encoder_hidden_states=encoder_last_hidden_state,output_hidden_states=True).hidden_states

        elif self.args.model_path in ['naver-clova-ix/donut-base-finetuned-docvqa1']:
            decoder_input_ids = sentences.to(self.args.device)
            decoder_input_ids = decoder_input_ids.unsqueeze(0) if decoder_input_ids.dim() == 1 else decoder_input_ids
            if decoder_input_ids.shape[0] > 1:
                decoder_input_ids = decoder_input_ids.transpose(0, 1)

            pixel_values = self.last_pixel_values.to(self.args.device)

            # 1. Encoder forward pass
            encoder_outputs = self.model.encoder(pixel_values=pixel_values)
            encoder_hidden_states = encoder_outputs.last_hidden_state


            # 2. Decoder embeddings
            token_embeds = self.model.decoder.get_input_embeddings()(decoder_input_ids)
            pos_embeds = self.model.decoder.model.decoder.embed_positions(decoder_input_ids)
            hidden_states = token_embeds + pos_embeds

            # Apply layer normalization if present
            if hasattr(self.model.decoder.model.decoder, 'layernorm_embedding'):
                hidden_states = self.model.decoder.model.decoder.layernorm_embedding(hidden_states)

            layer_inputs = []

            print("decoder_input_ids.shape:", decoder_input_ids.shape)
            print("pixel_values.shape:", pixel_values.shape)
            print("token_embeds.shape:", token_embeds.shape)
            print("pos_embeds.shape:", pos_embeds.shape)
            print("hidden_states.shape (init):", hidden_states.shape)
            print("encoder_hidden_states.shape:", encoder_hidden_states.shape)

            # 3. Forward through each decoder layer with the causal mask
            for i in range(layers):

                if encoder_hidden_states.size(0) != hidden_states.size(0):
                    encoder_hidden_states = encoder_hidden_states.expand(hidden_states.size(0), -1, -1)

                print(f"[{i}] hidden_states:", hidden_states.shape)

                decoder_layer = self.model.decoder.model.decoder.layers[i]
                hidden_states, *_ = decoder_layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    past_key_value=None,
                    use_cache=False,
                    output_attentions=False
                )
                layer_inputs.append(hidden_states.clone())

            return layer_inputs

    def is_bert(self):
        return self.args.model_path in ['bert-base-uncased']
    
    def is_decoder(self):
        return self.args.model_path in ['gpt2', 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf','openai-community/gpt2-large', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B',"naver-clova-ix/donut-base-finetuned-docvqa"]
        
    def has_rope(self):
        return self.args.model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']

    def has_bos(self):
        return self.start_token is not None
    def is_lower(self):
        return self.args.precision in ['8bit', 'half']

    def get_processor(self):
        return self.processor

    def get_model(self):
        return self.model

    def is_donut(self):
        return self.args.model_path in ['naver-clova-ix/donut-base-finetuned-docvqa']

    def set_last_pixel_values(self, pixel_values):
        self.last_pixel_values = pixel_values.to(self.args.device)



