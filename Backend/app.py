# views.py
import credentials, requests
import sys
import torch
from flask import Flask
from flask import render_template, request, session
from itertools import chain
from model.interact import sample_sequence,load_model
from model.config import params_setup, load_personality
from model.dataset import tokenize


# Initialize the app
app = Flask(__name__, instance_relative_config=True)

# Load the config file

app.config.from_object('config')
class Bot:
    def __init__(self,args,personality,history):
        self.args = args
        self.history = history
        filter_words = [". ! ? a an the"]
        #load model and its tokenizer
        self.model, self.tokenizer = load_model(self.args)
        self.personality = tokenize(personality,self.tokenizer)
        self.filter_words = tokenize(filter_words,self.tokenizer)

    def predict(self,raw_text):
        self.history.append(self.tokenizer.encode(raw_text))
        
        with torch.no_grad():
            out_ids = sample_sequence(self.args,self.personality, self.history, self.tokenizer, self.model)
            while True:
                out_ids = sample_sequence(self.args,self.personality, self.history, self.tokenizer, self.model) 
                #print("out_ids: ", out_ids)
                if not self.check_repetition_cross_turn(self.history,out_ids,self.filter_words):
                    break
                else:
                    print("long common: ", self.tokenizer.decode(out_ids, skip_special_tokens=True))
                if not out_ids: # avoid infinitely looping over special token
                    break
        self.history.append(out_ids)
        self.history = self.history[-(2*self.args.num_history+1):]
        out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return out_text
    
    def check_repetition_cross_turn(self,history,output,fil_words):
        if not output:
            return True
        for seq in history:
            seq = list(filter(lambda x: x not in chain(*fil_words), seq)) #filter puncation
            output = list(filter(lambda x: x not in chain(*fil_words), output)) #filter puncation
            if self.lcs(seq,output) >= self.args.longest_common:
                return True
        return False

    def lcs(self,X, Y): 
        # find the length of the strings 
        m = len(X) 
        n = len(Y) 
    
        # declaring the array for storing the dp values 
        L = [[None]*(n + 1) for i in range(m + 1)] 
        # print(L)
    
        """Following steps build L[m + 1][n + 1] in bottom up fashion 
        Note: L[i][j] contains length of LCS of X[0..i-1] 
        and Y[0..j-1]"""
        for i in range(m + 1): 
            for j in range(n + 1): 
                if i == 0 or j == 0 : 
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]: 
                    L[i][j] = L[i-1][j-1]+1
                else: 
                    L[i][j] = max(L[i-1][j], L[i][j-1]) 
    
        # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
        return L[m][n] 


#---------------------------
#   MESSENGER API
#---------------------------

# Adds support for GET requests to our webhook
@app.route('/webhook',methods=['GET'])
def webhook_authorization():
    verify_token = request.args.get("hub.verify_token")
    # Check if sent token is correct
    if verify_token == app.config['WEBHOOK_VERIFY_TOKEN']:
        # Responds with the challenge token from the request
        return request.args.get("hub.challenge")
    return 'Unable to authorise.'

@app.route("/webhook", methods=['POST'])
def webhook_handle():
    data = request.get_json()
    message = data['entry'][0]['messaging'][0]['message']
    print(message['text'], file=sys.stderr)
    bot_response = chatbot.predict(message['text'])
    sender_id = data['entry'][0]['messaging'][0]['sender']['id']
    if message['text']:
        request_body = {
                'recipient': {
                    'id': sender_id
                },
                'message': {"text":bot_response}
            }
        response = requests.post('https://graph.facebook.com/v8.0/me/messages?access_token='+app.config['TOKEN'],json=request_body).json()
        return response
    return 'ok'

#---------------------------
#   TESTING
#---------------------------

@app.route('/')
def index():
    #print(app.config['WEBHOOK_VERIFY_TOKEN'], file=sys.stderr)
    #print(session.get("personality"), file=sys.stderr)
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")

@app.before_first_request
def before_first_request():
    global chatbot
    args = params_setup()
    personality = load_personality(args)
    history = []
    chatbot = Bot(args, personality, history)

if __name__ == '__main__':
    app.secret_key = app.config['SECRET_KEY']
    app.run(threaded=True, port=80)