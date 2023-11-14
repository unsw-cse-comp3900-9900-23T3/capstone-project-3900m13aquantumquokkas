import http.server
import socketserver
from backend.backend import detect
import json
import os

port = 8000


class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/out":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            user_input = data.get("user_input")
            print(user_input)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            self.wfile.write(detect(user_input).encode("utf-8"))


Handler = MyHandler

with socketserver.TCPServer(("", port), Handler) as httpd:
    print(f"Serving at port {port}")
    httpd.serve_forever()


# to run, type python server.py and then open up http://localhost:8000/frontend/index.html
