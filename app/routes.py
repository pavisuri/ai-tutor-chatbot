from flask import request, render_template, jsonify
from chat import chat_with_user

def setup_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/get_answer', methods=['POST'])
    def get_answer():
        user_query = request.form['query']
        answer = chat_with_user(user_query)
        return jsonify({'response': answer})
    