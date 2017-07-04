import http.server
import socketserver
import sys
import subprocess
import json
from urllib.error import URLError
from urllib.request import urlretrieve
import urllib.parse


from evaluation import get_pred, init

PORT = 4000

MODEL_PATH='../savings/resnet_2017_6_29-18:42:50/model.h5'
IS_DECAF = False
K = 3

model = None
    
def query_predict(httpd, model, query):

    result = {}

    if not 'url' in query:
        msg = "Error: missing url argument in predict query"
        return httpd.respond_with_error(422, msg)

    url = query['url'][0]
    ressource = None

    try:
        print("Downloading URL " + url)
        urlretrieve(url, '/tmp/rasta_tmp')
        #print("FILE " + str(ressource))
    except URLError as err:
        resp = {"error": 200,
                "user_error_msg": "Cannot download ressource at url '{}': {}".format(url, err.reason) }
        return httpd.respond(resp)

    except ValueError as err:
        resp = {"error": 500,
                "user_error_msg": "Cannot download ressource at url '{}'. Invalid URL.".format(url, str(type(err))) }
        return httpd.respond(resp)
    except Exception as e:
        resp = {"user_error": 500,
                "user_error_msg": "Cannot download ressource at url '{}'. Exception {} occured.".format(url, str(type(e))) }
        return httpd.respond(resp)
    
    pred,pcts = get_pred(model, '/tmp/rasta_tmp', IS_DECAF, K)

    pcts = [ str(i) for i in pcts ]
    resp = { 'pred' : pred, 'pcts' : pcts, 'k' : K }

    return httpd.respond(resp)

class Handler(http.server.BaseHTTPRequestHandler):

    query_dispatcher = { 'predict' : query_predict }

    def respond(self, data):
        data['error'] = 200
        datastr = json.dumps(data).encode("utf-8")
        self.send_response(data['error'])
        self.send_header("Content-type", 'text/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(datastr)
	
    def respond_with_error(self, err, msg):
        print("Sending error: " + msg)
        datastr = json.dumps({ 'error' : err, 'error_msg' : msg }).encode("utf-8")
        self.send_response(err)
        self.send_header("Content-type", 'text/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(datastr)
	    


    def do_GET(self):
        req = urllib.parse.urlparse(self.path)
        msg = "Ok"
        query = None

        try:
            query = urllib.parse.parse_qs(req.query, strict_parsing = True)
        except ValueError as e:
            msg = "Error: cannot parse query: '" + req.query + "'"
            return self.respond_with_error(422, msg)

        if not 'type' in query:
            msg = "Error: no type specified in query: " + req.query
            return self.respond_with_error(422, msg)
        
        qtype = query['type'][0]
        if not qtype in Handler.query_dispatcher:
            msg = "Error: invalid query type: " + qtype
            return self.respond_with_error(422, msg)
        
        global model
        result = Handler.query_dispatcher[ qtype ] ( self, model, query )

def main():
    httpd = socketserver.TCPServer(("", PORT), Handler)

    global model
    model = init(MODEL_PATH, IS_DECAF) # todo outch

    print("serving at port", PORT)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
        sys.exit(0)


if __name__ == '__main__':
    main()
