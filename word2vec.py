import torch, torchvision 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from sklearn.feature_extraction.text import CountVectorizer
import pickle


class Vocab(CountVectorizer):
    r"""
    We use Count Vectorizer as a Tokenizer. A good extension will give us a Tokenizer with built-in id2word and word2id functions
    """
    def __init__(self, texts, **kwargs):
        super(Vocab, self).__init__(**kwargs)
        self.fit(texts)
        self.id2w = self.get_feature_names()

    def __len__(self):
        return len(self.vocabulary_)

    def _to_indices(self, sent):
        analyzer = self.build_analyzer()
        seq = analyzer(sent)
        indices = []
        for w in seq:
            try:
                indices.append(self.vocabulary_[w])
            except KeyError:
                continue
        return indices

    def __getitem__(self, id):
        return self.id2w[id]


def save_model(model, filename):
    print('saving model to %s ...' % filename)
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    print('loading model from %s ...' % filename)
    return pickle.load(open(filename, 'rb'))


CONTEXT_SIZE = 2


class MyWord2Vec(nn.Module):
    
    r"""
    Implement model CBOW
    """
    def __init__(self, vocab_size, embed_size):
        super(MyWord2Vec, self).__init__()
        self.embeds = nn.Embedding(vocab_size, embed_size)
        self.linear1 = nn.Linear(2 * CONTEXT_SIZE * embed_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)
    
    def forward(self, x):
        embeds = self.embeds(x).view((x.shape[0], -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def get_embeds(self, wordid):
        return self.embeds(torch.tensor([wordid], dtype=torch.long)).view(-1)


def word_embeds(vocab, w2v):
    for w in vocab.vocabulary_:
        yield w2v.get_embeds(vocab.vocabulary_[w])
        

if __name__ == '__main__':
    # load dataset 
    fname = 'data/new-york-times-articles/nytimes_news_articles.txt'
    articles = []
    errs = 0
    with open(fname, 'r') as f:
        art = []
        while True:
            line = f.readline()
            if not line:
                break
            if 'URL' in line:
                if art != []:
                    paper = ' '.join(art)
                    if 'No corrections appeared in print' not in paper: #error papers
                        articles.append(paper)
                    else:
                        errs += 1
                art = []
            elif line not in {'\n', ' '}:
                art.append(line.rstrip('\n'))

    # generate vocab
    vocab_size = 8000

    kwargs = {'stop_words': 'english', 'max_df': 0.5, 'min_df': 3, 'ngram_range': (1, 1),
            'max_features': vocab_size, 'strip_accents': 'unicode'}
    vocab = Vocab(articles, **kwargs)

    # preprocess data
    ctx = []
    trg = []
    for text in articles:
        raw_text = vocab._to_indices(text)
        for i in range(2, len(raw_text) - 2):
            context = [raw_text[i - 2], raw_text[i - 1],
                        raw_text[i + 1], raw_text[i + 2]]
            target = raw_text[i]
            ctx.append(context) 
            trg.append(target)
    
    ctx = torch.tensor(ctx, dtype=torch.long)
    trg = torch.tensor(trg, dtype=torch.long)

    # define our model
    epochs = 10
    embed_size = 50
    batch_size = 16

    model = MyWord2Vec(vocab_size, embed_size)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss_function = nn.NLLLoss()
    
    # training
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for idx in range(0, ctx.shape[0], batch_size):
            #for idx, (context, target) in enumerate(data):
            context_idxs = ctx[idx: idx+batch_size]

            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_function(log_probs, trg[idx: idx+batch_size])

            # backward pass and update the gradient
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if idx % 10 == 0:
                print('Ep {}/{} {}/{} loss={:.4f}'.format(epoch, epochs, idx, trg.shape[0], total_loss), flush=True, end='\r')
        losses.append(total_loss)
        print('\n Loss: %.4f' % losses)

