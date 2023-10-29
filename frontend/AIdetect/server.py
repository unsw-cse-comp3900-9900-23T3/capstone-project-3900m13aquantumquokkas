import http.server
import socketserver
from out import out

port = 8000

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/out':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(out().encode('utf-8'))

Handler = MyHandler

with socketserver.TCPServer(("", port), Handler) as httpd:
    print(f"Serving at port {port}")
    httpd.serve_forever()


# to run, type python server.py and then open up http://localhost:8000/index.html