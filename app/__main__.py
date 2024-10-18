from flask import Flask
from routes import setup_routes
import os


# Initialize Flask with custom template and static folder paths
app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Setup routes
setup_routes(app)

if __name__ == '__main__':
    app.run(debug=True)
