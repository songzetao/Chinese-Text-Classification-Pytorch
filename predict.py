from importlib import import_module
import os
import pickle as pkl
import oneflow as torch

#TODO:增加predict文件
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def process_data(title, config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    # print(f"Vocab size: {len(vocab)}")
    # print(vocab)
    config.n_vocab = len(vocab)
    def load_dataset(path, pad_size=32):
        contents = []
        # with open(path, 'r', encoding='UTF-8') as f:
        # for line in tqdm(f):

        lin = title.strip()
        content = lin
        words_line = []
        token = tokenizer(content)
        seq_len = len(token)
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        # word to id
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))
        contents.append((words_line, seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    # train = load_dataset(config.train_path, config.pad_size)
    # dev = load_dataset(config.dev_path, config.pad_size)
    # test = load_dataset(config.test_path, config.pad_size)
    # return vocab, train, dev, test
    return load_dataset(config.train_path, config.pad_size)



def test(config, model, test_iter):
    # test
    print('loading model...')
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif) 

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def predict(title):
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    with torch.no_grad():
        result = model(title)
    # print(result)
    print(torch.argmax(result[0]))

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    # embedding = 'embedding_SougouNews.npz'
    # if args.embedding == 'random':
    #     embedding = 'random'
    # model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    # if model_name == 'FastText':
    #     from utils_fasttext import build_dataset, build_iterator, get_time_dif
    #     embedding = 'random'
    # else:
    #     from utils import build_dataset, build_iterator, get_time_dif
    
    model_name = 'TextCNN'
    x = import_module('models.' + model_name)
    print(x)
    
    config = x.Config(dataset, 'random')
    
    # result = process_data('中华女子学院：本科层次仅1专业招男生', config,False)
    # print("result")
    # print(result)
    # title_to_idx = result[0][0]
    # # 数据处理完毕，进行模型测试
    # # test(config, model, test_iter)
    # title_pre = [torch.tensor([title_to_idx]).to(config.device)]
    # print(title_pre)
    # predict(title_pre)

    path = 'THUCNews/data/train.txt'

    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            title = line.split('\t')[0]
            print(title)
            result = process_data(title, config,False)
            # print("result")
            # print(result)
            title_to_idx = result[0][0]
            # 数据处理完毕，进行模型测试
            # test(config, model, test_iter)
            title_pre = [torch.tensor([title_to_idx]).to(config.device)]
            # print(title_pre)
            predict(title_pre)

