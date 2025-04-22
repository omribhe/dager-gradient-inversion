import datetime
import numpy as np
import torch
import evaluate
from transformers import DonutProcessor, VisionEncoderDecoderConfig

#omri branch test

from utils.models import ModelWrapper
from utils.data import TextDataset, DonutDataset
from utils.filtering_encoder import filter_encoder
from utils.filtering_decoder import filter_decoder
from utils.functional import get_top_B_in_span, check_if_in_span, remove_padding, filter_outliers, get_span_dists
from args_factory import get_args
import time

from scipy.optimize import linear_sum_assignment

# old seed: 100
args = get_args()
np.random.seed(args.rng_seed)
torch.manual_seed(args.rng_seed)

total_correct_tokens = 0
total_tokens = 0
total_correct_maxB_tokens = 0

def filter_l1(args, model_wrapper, R_Qs):
    tokenizer = model_wrapper.tokenizer
    res_pos, res_ids, res_types = [], [], []
        
    sentence_ends = []
    p = 0
    n_tokens = 0

    while True:
        print(f'L1 Position {p}')
        embeds = model_wrapper.get_embeddings(p)
        if model_wrapper.is_bert(): # or model_wrapper.is_donut():
            if args.defense_noise is None:
                _, res_ids_new, res_types_new = get_top_B_in_span(R_Qs[0], embeds, args.batch_size, args.l1_span_thresh, args.dist_norm)
            else:
                raise NotImplementedError
        else:
            if args.defense_noise is None:
                _, res_ids_new = get_top_B_in_span(R_Qs[0], embeds, args.batch_size, args.l1_span_thresh, args.dist_norm)
            else:
                std_thrs = args.p1_std_thrs if p==0 else None
                d = get_span_dists(args, model_wrapper, R_Qs, embeds, p)
                res_ids_new = filter_outliers(d, std_thrs=std_thrs, maxB=max(50*model_wrapper.args.batch_size, int(0.05*len(model_wrapper.tokenizer))))
            res_types_new = torch.zeros_like(res_ids_new)
        res_pos_new = torch.ones_like( res_ids_new ) * p
        
        del embeds
        
        res_types += [res_types_new.tolist()]
        ids = res_ids_new.tolist()
        if len(ids) == 0 or p > tokenizer.model_max_length or p > args.max_len:
            break
        while model_wrapper.eos_token in ids:
            end_token_ind = ids.index(model_wrapper.eos_token)
            sentence_token_type = res_types[-1][ end_token_ind ]
            sentence_ends.append((p,sentence_token_type))
            ids = ids[:end_token_ind] + ids[end_token_ind+1:]
            res_types[-1] = res_types[-1][:end_token_ind] + res_types[-1][end_token_ind+1:]
        res_ids += [ids]
        res_pos += res_pos_new.tolist()
        n_tokens += len(ids)
        p += 1
        if model_wrapper.has_rope():
            break
        
    return res_pos, res_ids, res_types, sentence_ends

def reconstruct(args, device, sample, metric, model_wrapper: ModelWrapper):
    global total_correct_tokens, total_tokens, total_correct_maxB_tokens

    tokenizer = model_wrapper.tokenizer
    if args.dataset == 'donut':
        pixel_values, orig_batch, true_labels, attention_mask = sample
        # Create attention mask
        orig_batch = {"pixel_values": pixel_values.unsqueeze(0),"input_ids": orig_batch.unsqueeze(0), 'attention_mask': attention_mask.unsqueeze(0)}
    else:
        sequences, true_labels = sample
        orig_batch = tokenizer(sequences,padding=True, truncation=True, max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),return_tensors='pt').to(args.device)
    if args.dataset == 'donut':
        true_grads, encoder_hidden_state = model_wrapper.compute_grads(batch = {"pixel_values": pixel_values.unsqueeze(0),"decoder_input_ids": orig_batch['input_ids']}, y_labels=true_labels.unsqueeze(0))
    else:
        true_grads = model_wrapper.compute_grads(orig_batch.to(args.device), true_labels.to(args.device))
    if args.defense_noise is not None:
        for grad in true_grads:
            grad.data = grad.data + torch.randn(grad.shape) * args.defense_noise
    prediction, predicted_sentences, predicted_sentences_scores = [], [], []
    #import pdb;pdb.set_trace() 
    with torch.no_grad():
        B, R_Qs = model_wrapper.get_matrices_expansions(true_grads, B=None, tol=args.rank_tol)
        R_Q = R_Qs[0]
        R_Q2 = R_Qs[1]
        
        if B is None:
            reference = []
            for i in range(orig_batch['input_ids'].shape[0]):
                reference += [remove_padding(tokenizer, orig_batch['input_ids'][i], left=(args.pad=='left'))]
            return ['' for _ in range(len(reference))], reference
        R_Q, R_Q2 = R_Q.to(args.device), R_Q2.to(args.device)
        total_true_token_count, total_true_token_count2 = 0, 0
        for i in range( orig_batch['input_ids'].shape[1] ):
            total_true_token_count2 += args.batch_size - ( orig_batch['input_ids'][:,i] == model_wrapper.pad_token).sum()
            uniques = torch.unique(orig_batch['input_ids'][:,i])
            total_true_token_count += uniques.numel()
            if model_wrapper.pad_token in uniques.tolist():
                total_true_token_count -= 1
   
        print(f"{B}/{total_true_token_count}/{total_true_token_count2}")
        if args.neptune:
            args.neptune['logs/max_rank'].log( B )
            args.neptune['logs/batch_tokens'].log( total_true_token_count2 ) 
            args.neptune['logs/batch_unique_tokens'].log( total_true_token_count )
         
        del true_grads 
       
        res_pos, res_ids, res_types, sentence_ends = filter_l1(args, model_wrapper, R_Qs)
        
        print( orig_batch )
        print( orig_batch['input_ids'].T )
        if len(res_ids) == 0:        
            reference = []
            for i in range(orig_batch['input_ids'].shape[0]):
                reference += [remove_padding(tokenizer, orig_batch['input_ids'][i], left=(args.pad=='left'))]        
            return ['' for _ in reference], reference
        if len(res_ids[0])<100000:
            print( res_pos, res_ids, res_types )
        
        rec_l1, rec_l1_maxB, rec_l2 = [], [], []

        for s in range( orig_batch['input_ids'].shape[0] ):
            sentence_in = True
            sentence_in_max_B = True
            orig_sentence = orig_batch['input_ids'][s]
            last_idx = torch.where(orig_batch['input_ids'][s] != tokenizer.pad_token_id)[0][-1].item()
            for pos, token in enumerate( orig_sentence ):
                if not model_wrapper.is_bert() and pos == last_idx:
                    break
                if pos >= len(res_ids) and not model_wrapper.has_rope():
                    sentence_in = False
                    break
                if token == model_wrapper.pad_token and args.pad=='right':
                    pos-=1
                    break
                elif token == model_wrapper.pad_token and args.pad=='left':
                    continue
                if model_wrapper.has_rope():
                    total_correct_tokens += 1 if token in res_ids[0] else 0
                    total_correct_maxB_tokens += 1 if token in res_ids[0][:min(args.batch_size, len(res_ids[0]))] else 0
                    total_tokens += 1
                else:
                    total_correct_tokens += 1 if token in res_ids[pos] else 0
                    total_correct_maxB_tokens += 1 if token in res_ids[pos][:min(args.batch_size, len(res_ids[pos]))] else 0   
                    total_tokens += 1
                if token == model_wrapper.eos_token and args.pad=='right':
                    break
                
                if model_wrapper.has_rope():
                    if model_wrapper.has_bos() and token==model_wrapper.start_token:
                        continue
                    sentence_in = sentence_in and (token in res_ids[0])
                    sentence_in_max_B = sentence_in_max_B and (token in res_ids[0][:min(args.batch_size, len(res_ids[0]))])
                else:
                    sentence_in = sentence_in and (token in res_ids[pos]) 
                    sentence_in_max_B = sentence_in_max_B and (token in res_ids[pos][:min(args.batch_size, len(res_ids[pos]))]) 
            if model_wrapper.is_bert():
                sentence_in = sentence_in and (pos, orig_batch['token_type_ids'][s][pos]) in sentence_ends
                sentence_in_max_B = sentence_in and (pos, orig_batch['token_type_ids'][s][pos]) in sentence_ends

            rec_l1.append( sentence_in )
            rec_l1_maxB.append( sentence_in_max_B )
            if model_wrapper.has_rope():
                orig_sentence = (orig_sentence).reshape(1,-1)
            else:
                orig_sentence = (orig_sentence[:pos+1]).reshape(1,-1)
            if model_wrapper.is_bert():
                token_type_ids = (orig_batch['token_type_ids'][s][:orig_sentence.shape[1]]).reshape(1,-1)
                input_layer1 = model_wrapper.get_layer_inputs(orig_sentence, token_type_ids)[0]
            else:
                attention_mask = orig_batch['attention_mask'][s][:orig_sentence.shape[1]].reshape(1,-1)
                if args.dataset == 'donut':
                    input_layer1 = model_wrapper.get_layer_inputs(orig_sentence, attention_mask=attention_mask, encoder_last_hidden_state=encoder_hidden_state)[0]
                else:
                    input_layer1 = model_wrapper.get_layer_inputs(orig_sentence, attention_mask=attention_mask)[0]

            sizesq2 = check_if_in_span(R_Q2, input_layer1, args.dist_norm)
            boolsq2 = sizesq2 < args.l2_span_thresh
            print( sizesq2 )
        
            if args.task == 'next_token_pred':
                rec_l2.append( torch.all(boolsq2[:-1]).item() )
            elif model_wrapper.has_rope(): 
                rec_l2.append( torch.all(boolsq2[1:]).item() )
            else:
                rec_l2.append( torch.all(boolsq2).item() )
        
        print( f'Rec L1: {rec_l1}, Rec L1 MaxB: {rec_l1_maxB}, Rec MaxB Token: {total_correct_maxB_tokens/total_tokens}, Rec Token: {total_correct_tokens/total_tokens}, Rec L2: {rec_l2}' )

        if args.neptune:
            args.neptune['logs/rec_l1'].log( np.array(rec_l1).sum() )
            args.neptune['logs/rec_l1_max_b'].log( np.array(rec_l1_maxB).sum() ) 
            args.neptune['logs/maxB token'].log( total_correct_maxB_tokens/total_tokens ) 
            args.neptune['logs/token'].log( total_correct_tokens/total_tokens ) 
            args.neptune['logs/rec_l2'].log( np.array(rec_l2).sum() ) 
            
        if model_wrapper.is_decoder():
            max_ids = -1
            for i in range(len(res_ids)):
                if len(res_ids[i]) > args.max_ids:
                    max_ids = args.max_ids
            if args.dataset == 'donut':
                encoder_last_hidden_state = encoder_hidden_state
            else:
                encoder_last_hidden_state = None
            predicted_sentences, predicted_sentences_scores, top_B_incorrect_sentences, top_B_incorrect_scores  = filter_decoder(args, model_wrapper, R_Qs, res_ids, max_ids=max_ids, encoder_last_hidden_state=encoder_last_hidden_state)
            if len(predicted_sentences) < orig_batch['input_ids'].shape[0]:
                predicted_sentences += top_B_incorrect_sentences
                predicted_sentences_scores += top_B_incorrect_scores
        else:
            for l,token_type in sentence_ends:
                
                if args.l1_filter == 'maxB':
                    max_ids = args.batch_size
                elif args.l1_filter == 'all':
                    max_ids = -1
                else:
                    assert False

                if args.l2_filter == 'non-overlap':
                    correct_sentences = []
                    approx_sentences = []
                    approx_scores = []
                    for sent, sc in zip( predicted_sentences, predicted_sentences_scores ): 
                        if sc < args.l2_span_thresh:
                            correct_sentences.append( sent )
                        else:
                            approx_sentences.append( sent )
                            approx_scores.append( sc )

                    new_predicted_sentences, new_predicted_scores = filter_encoder(args, model_wrapper, R_Q2, l, token_type, res_ids, correct_sentences, approx_sentences, approx_scores, max_ids, args.batch_size)
                elif args.l2_filter == 'overlap':
                    new_predicted_sentences, new_predicted_scores = filter_encoder(args, model_wrapper, R_Q2, l, token_type, res_ids, [], [], [], max_ids, args.batch_size)
                else:
                    assert False

                predicted_sentences += new_predicted_sentences
                predicted_sentences_scores += new_predicted_scores

    correct_sentences = []
    approx_sentences = []
    approx_sentences_ext = []
    approx_sentences_lens = []
    approx_scores = []
    max_len = max( [len(s) for s in predicted_sentences] )
    for sent, sc in zip( predicted_sentences, predicted_sentences_scores ): 
        if sc < args.l2_span_thresh:
            correct_sentences.append( sent )
        else:
            approx_sentences.append( sent )
            approx_sentences_ext.append( sent + [-1]*(max_len - len(sent)) )
            approx_sentences_lens.append( len(sent) )
            approx_scores.append( sc )
    approx_scores = torch.tensor(approx_scores)
    approx_sentences_lens = torch.tensor(approx_sentences_lens)

    if len(approx_sentences) > 0:
        for i in range(len(correct_sentences)):
            sent = correct_sentences[i]
            similar_sentences = (torch.tensor(sent) == torch.tensor(approx_sentences_ext)[:,:len(sent)]).sum(1) >= torch.min(approx_sentences_lens,torch.tensor(len(sent)))*args.distinct_thresh
            approx_scores[similar_sentences] = torch.inf
        
        predicted_sentences = correct_sentences.copy()
        for i in range(len(correct_sentences), args.batch_size):
            idx = torch.argmin( approx_scores )
            predicted_sentences.append( approx_sentences[idx] )
            similar_sentences = (torch.tensor(approx_sentences_ext[idx]) == torch.tensor(approx_sentences_ext)).sum(1) >= max_len*args.distinct_thresh
            approx_scores[similar_sentences] = torch.inf

    for s in predicted_sentences:
        prediction.append( tokenizer.decode(s) )
    if args.neptune:
        args.neptune['logs/num_pred'].log( len(correct_sentences) ) 
    
    reference = []
    for i in range(orig_batch['input_ids'].shape[0]):
        reference += [remove_padding(tokenizer, orig_batch['input_ids'][i, :tokenizer.model_max_length], left=(args.pad=='left'))]
    if len(prediction) > len(reference):
        prediction = prediction[:len(reference)]

    if model_wrapper.is_decoder():
        new_prediction = []
        og_side = tokenizer.padding_side
        tokenizer.padding_side='right'
        for i in range(len(reference)):
            sequences = [reference[i]] + prediction
            batch = tokenizer(sequences,padding=True, truncation=True, return_tensors='pt')
            best_idx = (batch['input_ids'][1:] == batch['input_ids'][0]).sum(1).argmax()
            new_prediction.append(prediction[best_idx])
        tokenizer.padding_side=og_side
        prediction=new_prediction
    else:
        cost = np.zeros((len(prediction), len(prediction)))
        for i in range(len(prediction)):
            for j in range(len(prediction)):
                # fm = metric.compute(predictions=[prediction[i]], references=[reference[j]])['rouge1'].mid.fmeasure
                fm = metric.compute(predictions=[prediction[i]], references=[reference[j]])['rouge1']
                cost[i, j] = 1.0 - fm
        row_ind, col_ind = linear_sum_assignment(cost)

        ids = list(range(len(prediction)))
        ids.sort(key=lambda i: col_ind[i])
        new_prediction = []
        for i in range(len(prediction)):
            new_prediction += [prediction[ids[i]]]
        prediction = new_prediction

    return prediction, reference


def print_metrics(args, res, suffix):
    #sys.stderr.write(str(res) + '\n')
    for metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
        curr = res[metric].mid
        print(f'{metric:10} | fm: {curr.fmeasure*100:.3f} | p: {curr.precision*100:.3f} | r: {curr.recall*100:.3f}', flush=True)
        if args.neptune:
            args.neptune[f'logs/{metric}-fm_{suffix}'].log(curr.fmeasure*100)
            args.neptune[f'logs/{metric}-p_{suffix}'].log(curr.precision*100)
            args.neptune[f'logs/{metric}-r_{suffix}'].log(curr.recall*100)
    sum_12_fm = res['rouge1'].mid.fmeasure + res['rouge2'].mid.fmeasure
    if args.neptune:
        args.neptune[f'logs/r1fm+r2fm_{suffix}'].log(sum_12_fm*100)
    print(f'r1fm+r2fm = {sum_12_fm*100:.3f}', flush=True)
    print()

def main():
    device = torch.device(args.device)
    metric = evaluate.load('rouge')
    model_wrapper = ModelWrapper(args)

    if args.dataset == 'donut':

        dataset = DonutDataset(
            dataset_name_or_path='nielsr/docvqa_1200_examples_donut',
            max_length=128,
            split='train',
            task_start_token='<s_docvqa>',
            prompt_end_token='<s_answer>',
            sort_json_key=False,  # cord dataset is preprocessed, so no need for this
            image_size=[1280, 960],
            processor=model_wrapper.get_processor(),
            model=model_wrapper.get_model(),
        )
    else:
        dataset = TextDataset(args.device, args.dataset, args.split, args.n_inputs, args.batch_size, args.cache_dir)

    print('\n\nAttacking..\n', flush=True)
    predictions, references = [], []
    t_start = time.time()
    
    for i in range(args.start_input, min(args.n_inputs, args.end_input)):
        t_input_start = time.time()
        sample = dataset[i] # (seqs, labels)

        print(f'Running input #{i} of {args.n_inputs}.', flush=True)
        if args.neptune:
            args.neptune['logs/curr_input'].log(i)

        print('reference: ', flush=True)
        for seq in sample[0]:
            print('========================', flush=True)
            print(seq, flush=True)

        print('========================', flush=True)
        
        prediction, reference = reconstruct(args, device, sample, metric, model_wrapper)
        predictions += prediction
        references += reference

        print(f'Done with input #{i} of {args.n_inputs}.', flush=True)
        print('reference: ', flush=True)
        for seq in reference:
            print('========================', flush=True)
            print(seq, flush=True)
        print('========================', flush=True)

        print('predicted: ', flush=True)
        for seq in prediction:
            print('========================', flush=True)
            print(seq, flush=True)
        print('========================', flush=True)

        print('[Curr input metrics]:', flush=True)
        res = metric.compute(predictions=prediction, references=reference)
        # print_metrics(args, res, suffix='curr')

        print('[Aggregate metrics]:', flush=True)
        res = metric.compute(predictions=predictions, references=references)
        # print_metrics(args, res, suffix='agg')

        input_time = str(datetime.timedelta(seconds=time.time() - t_input_start)).split(".")[0]
        total_time = str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0]
        print(f'input #{i} time: {input_time} | total time: {total_time}', flush=True)
        print()
        print()

    print('Done with all.', flush=True)
    if args.neptune:
        args.neptune['logs/curr_input'].log(args.n_inputs)

if __name__ == '__main__':
    main()
