from flask import Flask, request, jsonify, render_template
import pickle
import base64
from io import BytesIO
from matplotlib.figure import Figure
from methods import *

app = Flask(__name__)

#with open('resources/model_pkl', 'rb') as file:
#    model=pickle.load(file)
#df1 = pd.read_csv('resources/df_to_predict.csv')

@app.route('/')
def home():
    return render_template('index2.html',pred_img='no')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [x for x in request.form.values()]
    #print(x_test) #exp of x_test : [['Tunisia', '2020-01-10', '2222-02-22']]
    '''
    A = x_test[1]
    B = x_test[2]
    A = datetime.date(int(A.split('-')[0]),int(A.split('-')[1]),int(A.split('-')[2]))
    B = datetime.date(int(B.split('-')[0]),int(B.split('-')[1]),int(B.split('-')[2]))
    
    l_days = pd.date_range(A,B)
    
    list_preds = predict_cases(A,B,df1,l_days,model)
    '''
    
    l_days = [1,2,3,4]
    list_preds = [1,2,3,4]
    
    fig = Figure()
    ax = fig.subplots()
    ax.plot(l_days,list_preds)
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    fig.savefig('foo', format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return render_template('index2.html',pred_img = data )
if __name__ == "__main__":
    app.run(debug=True)