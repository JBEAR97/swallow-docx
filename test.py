from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    import os
    return f'<h1>Docker OK! Files: {len(os.listdir("./DOCX"))}</h1>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
