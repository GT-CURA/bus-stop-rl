from flask import Flask, render_template
import logging 

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True
@app.route('/')
def index():
    return render_template('index.html')

def start_server(port=5000):
    """
    Starts the Flask app in a thread so it doesn't block.
    """
    import threading

    def run_app():
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
