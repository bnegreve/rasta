import http.server
import socketserver
import sys
import subprocess
import json
from urllib.error import URLError
from urllib.request import urlretrieve
import urllib.parse
from PIL import Image


from evaluation import get_pred, init

PORT = 4000

MODEL_PATH='../savings/best/model.h5'
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
        msg = "Cannot download ressource at url '{}': {}.".format(url, err.reason) 
        return httpd.respond_with_user_error(1, msg)
    except ValueError as err:
        msg = "Cannot download ressource at url '{}'. Invalid URL.".format(url, str(type(err))) 
        return httpd.respond_with_user_error(2, msg)
    except Exception as e:
        msg = "Cannot download ressource at url '{}'. Exception {} occured.".format(url, str(type(e)))
        return http.respond_with_error(self, 500, msg)

    try:
        img = Image.open('/tmp/rasta_tmp')
        img.close()
    except IOError:
        msg = "Url does not point to a supported image file."
        return httpd.respond_with_user_error(3, msg)

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
        print("Response ({}): ".format(datastr));
        return 200
	
    def respond_with_error(self, err, msg):
        print("Sending error: " + msg)
        datastr = json.dumps({ 'error' : err, 'error_msg' : msg }).encode("utf-8")
        self.send_response(err)
        self.send_header("Content-type", 'text/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(datastr)
        print("Response (Error!: {}): {}".format(code, datastr))
        return err

    def respond_with_user_error(self, user_err_code, user_err_msg):
        resp = {"user_error": user_err_code,
                "user_error_msg": user_err_msg }
        return self.respond(resp)



    def do_GET(self):
        req = urllib.parse.urlparse(self.path)
        msg = "Ok"
        query = None

        try:
            query = urllib.parse.parse_qs(req.query, strict_parsing = True)
        except ValueError as e:
            msg = "Error: cannot parse query: '" + req.query + "'"
            return self.respond_with_error(422, msg)

        if 'remote_addr' in query:
            print('Handling proxy query from ' + query['remote_addr'])

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
