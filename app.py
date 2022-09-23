from flask import Flask, render_template, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import numpy as np
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from utils import RNNClassifier

app = Flask(__name__)
app.config['SECRET_KEY'] = "Rajarshi"

class CommentForm(FlaskForm):
    comment = StringField("Type or paste the text below:", validators = [DataRequired()])
    submit = SubmitField("Submit")

vocab = torch.load("vocab.pt")
vocab_size = len(vocab)
embed_len = 50
hidden_dim = 128
n_layers=2
vocab = torch.load("vocab.pt")
class RNNClassifier(nn.Module):
    def __init__(self):
        super(RNNClassifier, self).__init__()
        self.seq = nn.Sequential(nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_len))
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_len)
        self.lstm = nn.LSTM(input_size = embed_len, hidden_size = hidden_dim, num_layers = n_layers, batch_first = True, bidirectional = True)
        #self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(2 * hidden_dim, 128),
                                    nn.ReLU(),
                                    # nn.Linear(128, 256),
                                    # nn.ReLU(),
                                    # nn.Linear(256, 128),
                                    # nn.ReLU(),
                                    nn.Linear(128, 6),
                                    nn.Sigmoid())
        

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.lstm(embeddings, (torch.randn(2 * n_layers, len(X_batch), hidden_dim, device = "cuda"), torch.randn(2 * n_layers, len(X_batch), hidden_dim, device = "cuda")))
        return self.linear(output[:,-1])

tokenizer = get_tokenizer('basic_english')
model = RNNClassifier()
model = model.to("cuda")
model.load_state_dict(torch.load("model.pth"))
vocab = torch.load("vocab.pt")
max_words = 2000
classes = np.array(["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"])
@app.route('/', methods = ['GET', 'POST'])
def showform():
    comment = None
    report = None
    pred = None
    form = CommentForm()
    if form.validate_on_submit():
        comment = form.comment.data
        form.comment.data = ""
        text_data = vocab(tokenizer(comment))
        text_data = text_data +([0]* (max_words-len(text_data))) if len(text_data)<max_words else text_data[:max_words]
        text_data = torch.tensor(text_data, dtype = torch.int32)
        text_data = torch.unsqueeze(text_data, 0)
        text_data = text_data.to("cuda")
        pred = torch.squeeze(model(text_data)).cpu().detach().numpy()
        pred = np.where(pred>0.5)
        # report.append("Toxic: {}".format(pred[0]))
        # report.append("Severe Toxic: {}".format(pred[1]))
        # report.append("Obscene: {}".format(pred[2]))
        # report.append("Threat: {}".format(pred[3]))
        # report.append("Insult: {}".format(pred[4]))
        # report.append("Identity Hate: {}".format(pred[5]))
        report = list(classes[pred])
        if(len(report) == 0):
            report = ["Non-Toxic"]
        flash("Report generated successfully!! Click on home if you want to get another prediction.")
    return render_template("index.html", report = report, form = form)
'''
@app.errorhandler(500)
def page_not_found(e):
    return render_template("500.html"), 500
'''
if __name__ == '__main__':
    app.run(debug=True)