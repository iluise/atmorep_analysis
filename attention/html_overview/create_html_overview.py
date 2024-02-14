from flask import Flask, render_template

model_id = ['bkkrobb9']
field = ['velocity_u']
batch_range = 14
heads_range = 16
app = Flask(import_name = __name__) #, static_folder='C://Users/iluise/cernbox/AtmoRep/code/atmorep/attention_plots/') #(__name__)

@app.route('/')
def home():
    return render_template('index.html', m_id=model_id, f = field, batch_range = batch_range,  heads_range = heads_range)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Allow connections from any IP address, use port 5001
