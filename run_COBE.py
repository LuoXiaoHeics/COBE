# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
from cProfile import label
import gc
from http.client import ImproperConnectionState
import os
from numpy.core.fromnumeric import argsort
from py import process
import torch
import logging
import random
import numpy as np
# from torch._C import half
from model import *
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer, WEIGHTS_NAME
from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from tensorboardX import SummaryWriter
# from model import *
import glob
import json
import shutil
import re
from glue_utils import *
import faiss

logger = logging.getLogger(__name__)
try:
    from apex import amp
except ImportError:
    amp = None
# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig)), ())
ALL_MODELS = (
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
    'bert-base-chinese',
    'bert-base-german-cased',
    'bert-large-uncased-whole-word-masking',
    'bert-large-cased-whole-word-masking',
    'bert-large-uncased-whole-word-masking-finetuned-squad',
    'bert-large-cased-whole-word-masking-finetuned-squad',
    'bert-base-cased-finetuned-mrpc',
    'bert-base-german-dbmdz-cased',
    'bert-base-german-dbmdz-uncased',
    'xlnet-base-cased',
    'xlnet-large-cased'
)



MODEL_CLASSES = {
    'COBE' : (BertConfig, BertCon, BertTokenizer),
}




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='model_output')
    parser.add_argument("--data_dir", default='data/fdu-mtl/', type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='bert_sent', type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument('--train_domains', nargs='+',required=True,)
    parser.add_argument('--test_domains', nargs='+', required=True, )
    parser.add_argument("--fix_tfm", default=0, type=int, help="whether fix the transformer params or not")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run testing.")
    parser.add_argument("--case_study", action='store_true',
                        help="Whether to run testing.")
    parser.add_argument("--do_dev", action='store_true',
                        help="Whether to run dev.")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--scheduler", default="linear", type=str, choices=["linear", "constant", "inv_sqrt"])
    parser.add_argument("--optimizer", default="adam", type=str, choices=["adam", "adafactor"])
    parser.add_argument('--seed', type=int, default=3250,
                        help="random seed for initialization")
    parser.add_argument('--knn_num', type=int, default=3,
                        help="number of knn")
    parser.add_argument("--no_cuda", action='store_true', default=False,
                        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Distributed learning")
    parser.add_argument('--fp16', default=False, action="store_true")
    args = parser.parse_args(["--data_dir", "data/fdu-mtl/", "--model_type", "COBE",
                              "--train_domains", "music",
                              "--test_domains",'books' ,
                              '--model_name_or_path', 'bert-base-uncased',
                              "--do_lower_case",'--do_test','--do_train',
                              '--output_dir','output'])
    # args = parser.parse_args()
    if args.fp16 and amp is None:
        print("No apex installed, fp16 not used.")
        args.fp16 = False
    return args


def train(args, train_dataset,tokenizer,domain_schema, model):
    """ Train the model """
    tb_writer = SummaryWriter(args.output_dir)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # del train_dataset
    # gc.collect()

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params'      : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.optimizer == "adam":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer == "adafactor":
        optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False,
                              relative_step=False)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
    elif args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    else:
        scheduler = get_inverse_sqrt_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)  # For reproducibility (even between python 2 and 3)

    # for n, p in model.bert.named_parameters():
    #     p.requires_grad = False
    # should_stop = False

    for epoch in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader,desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            loss = torch.tensor(0, dtype=float).to(args.device)

            inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              # XLM don't use segment_ids
                          'sent_labels': batch[3],
                          'meg': 'train'}
            l  = model(**inputs)

            if args.n_gpu >1:
                loss += l.mean()
            else :
                loss += l
            # loss+=1e-3*l2_reg
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if global_step % args.logging_steps == 0 or global_step == 1:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("epoch: {:d}, step: {:d}, "
                                "loss: {:.4f}, lr: {:g}".format(epoch, global_step,
                                                                (tr_loss - logging_loss) / args.logging_steps,
                                                                scheduler.get_lr()[0]))
                    logging_loss = tr_loss
                if 0 < args.max_steps < global_step:
                    should_stop = True
        if should_stop:
            break
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    trained_features = save_hiddens(args,train_dataloader,model)
    del train_dataloader
    tb_writer.close()
    return trained_features

def save_hiddens(args,data_loader,model):
    cls_hidden_list, label_list,dom_list = [],[],[]
    model.eval()
    for batch in data_loader:
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            eval_loss = torch.tensor(0, dtype=float).to(args.device)
            inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              # XLM don't use segment_ids
                              'sent_labels': batch[3],
                              'meg':'source'}
            cls_hidden  = model(**inputs)
            cls_hidden_list.append(cls_hidden)
            label_list.append(batch[3])
            dom_list.append(batch[5])
    cls_hidden_list = torch.cat(cls_hidden_list, axis=0)
    labels =torch.cat(label_list, axis=0)
    dom_list = torch.cat(dom_list, axis=0)
    save_data = TensorDataset(cls_hidden_list,labels,dom_list)
    torch.save(save_data,args.output_dir+'/hiddens')
    return save_data

def load_and_cache_examples(args, tokenizer, domain_schema, mode='train',task_name=None,half = [0]):

    # repalce with SentProcessor2 for 4 domains dataset
    processor = SentProcessor()

    # Load data features from cache or dataset file
    task = ""
    if not mode == 'evaluate':
        for a in domain_schema.keys():
            task += a
    else :
        task = task
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
            mode,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task),str(half)))

    if os.path.exists(cached_features_file):
        print("cached_features_file:", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == 'train':
            examples = processor.get_train_examples(args.data_dir, domain_schema)
        elif mode == 'test':
            examples = processor.get_test_examples(args.data_dir, domain_schema)
        elif mode == 'all':
            examples = processor.get_all_examples(args.data_dir, domain_schema)
        else:
            raise Exception("Invalid data mode %s..." % mode)
        features = convert_examples_to_features(examples=examples, max_seq_length=args.max_seq_length, tokenizer=tokenizer)

        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    del features 
    gc.collect()
    return dataset

# L2 
def estab_faiss(hidden):
    print("Index establishing: ...")
    index = faiss.IndexFlatL2(192)
    gpu_index = faiss.index_cpu_to_all_gpus(index)
    # 建立索引
    gpu_index.add(np.array(hidden))
    print(gpu_index.is_trained)
    return gpu_index

# cosine
def estab_faiss_cos(hidden):
    print("Index establishing: ...")
    index = faiss.IndexFlat(192, faiss.METRIC_INNER_PRODUCT)
    gpu_index = faiss.index_cpu_to_all_gpus(index)
    # 建立索引
    # gpu_index.add(np.array(hidden))
    gpu_index.add(np.array([h for h in hidden]))
    print(gpu_index.ntotal)
    print(gpu_index.is_trained)
    return gpu_index

# knn
def knn_interpolate_label_cos(args,h,index,temperature,gold_labels):
    logits = []
    for j in range(h.shape[0]):
        h_j = h[j].unsqueeze(0).detach().cpu().numpy()
        # h_j = h_j/np.sqrt(np.sum(h_j**2))
        D, I = index.search(h_j, args.knn_num)
        I = I[0]
        D= D[0]*temperature
        D = np.exp(D)
        D = D/np.sum(D)
        k_label = torch.zeros(2).to(args.device)
        for i in range(len(I)):
            k_label[gold_labels[I[i]]] += D[i]
        logits.append((np.array(k_label.detach().cpu().numpy())))
    return logits

def compute_metrics_absa(preds, labels):
    correct = 0
    errors = []
    for i in range(len(preds)):
        if labels[i] == np.argmax(preds[i]):
            correct+=1
        else:errors.append(i)
    accuracy = correct/len(labels)
    return accuracy,errors

# import visualize

def evaluate(args, model, tokenizer, mode, test_domain_schema,index,gold_labels):
    eval_dataset = load_and_cache_examples(args, tokenizer, test_domain_schema, mode=mode)

    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataloader)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    cls_hidden_list, label_list,dom_list = [],[],[]
    model.eval()
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            eval_loss = torch.tensor(0, dtype=float).to(args.device)
            inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              'sent_labels': batch[3],
                              'meg':'source'}
            cls_hidden  = model(**inputs)
            cls_hidden_list.append(cls_hidden)
            label_list.append(batch[3])
            # dom_list.append(batch[5])

    cls_hidden_list = torch.cat(cls_hidden_list, axis=0)
    labels =torch.cat(label_list, axis=0)
    dom_list = torch.cat(dom_list,axis=0)

    preds = knn_interpolate_label_cos(args,cls_hidden_list,index,temperature=5,gold_labels=gold_labels)
    result,errors = compute_metrics_absa(preds, labels)

    return result,eval_loss

def load_adv_examples(args,tokenizer):
    exampels = []
    with open('16_new.txt', 'r', encoding='ISO-8859-2') as inf:
        line = inf.readline().strip()
        while (line):
            label, word = line.split('\t')
            exampels.append(InputExamples(word, int(label), 0))
            line = inf.readline().strip()
    features = convert_examples_to_features(examples=exampels, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def main():
    args = init_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Setup CUDA
    torch.cuda.set_device(1)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu =1
    
    print("GPU number is : ~~~~~~~~~~~~~~~~  "+ str(args.n_gpu))
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    set_seed(args)

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          output_hidden_states=True)
    # config.aspect_num = args.aspect_num
    config.sent_number = 2
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case, cache_dir='./cache')

    if args.do_train:
        # Training
        domain_schema = get_domain_schema(args.train_domains)

        config.domain_number = len(domain_schema.keys())

        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config, cache_dir='./cache')
        model.to(args.device)

        if args.n_gpu >1:
            model = torch.nn.DataParallel(model)
        train_dataset = load_and_cache_examples(args, tokenizer, mode='train', domain_schema=domain_schema)
        trained_features = train(args, train_dataset,tokenizer,domain_schema, model)
        del train_dataset 
        gc.collect()
        del model
        torch.cuda.empty_cache()

    if args.do_test:
        # model = model_class.from_pretrained(args.output_dir)
        # model.to(args.device)
        # domain_schema = get_domain_schema(args.train_domains)
        # train_dataset = load_and_cache_examples(args, tokenizer, mode='train', domain_schema=domain_schema)
        # train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        # trained_features = save_hiddens(args,train_dataloader,model)
        if not args.do_train:
            trained_features = torch.load(args.output_dir+'/hiddens')
        hidden = [u[0].cpu().numpy() for u in trained_features]
        gold_labels = [u[1].cpu().numpy() for u in trained_features]
        index = estab_faiss_cos(hidden)
        args.model_type = args.model_type.lower()
        r = 0
        dom_id = 0
        with open (args.output_dir+'/test_results.txt','w') as f:
            for a in args.test_domains:
                domain_schema = {}
                domain_schema[a] = dom_id
                model = model_class.from_pretrained(args.output_dir)
                model.to(args.device)
                results,l = evaluate(args, model, tokenizer, 'test', domain_schema,index,gold_labels)
                print(a, results)
                r+=results
                f.write(a+": "+str(results)+'\n')
                dom_id+=1
            f.write('\n')
            f.write("Avg "+str(r/len(args.test_domains))+'\n')
        del model
        torch.cuda.empty_cache()
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
