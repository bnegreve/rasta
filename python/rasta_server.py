import http.server
import socketserver
import sys
import subprocess
import json
from urllib.error import URLError
from urllib.request import urlopen
from urllib.parse import unquote, quote, urlparse, parse_qs, urlunparse
from PIL import Image
from os import stat
from socket import timeout
from evaluation import get_pred, init

PORT = 4000

MODEL_PATH='../savings/best/model.h5'
IS_DECAF = False
K = 3
TIMEOUT=10
MAX_FILE_SIZE=10 * 1024 * 1024 # 10 MB
USER_AGENT_STRING="Mozilla/5.0 (X11; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0"


model = None
 
# #dummy functs (just for testing without having to load the model)
# def get_pred(a,b,c,d):
#     return ([], [])
# def init(model, is_decaf):
#     return None


   
def query_predict(httpd, model, query):

    result = {}

    if not 'url' in query:
        msg = "Error: missing url argument in predict query"
        return httpd.respond_with_error(422, msg)

    # quote the path of the image url because urlopen does not accept utf-8 paths (afaii)
    imgurl = urlparse(query['url'][0])
    imgurl = urlunparse((imgurl.scheme, imgurl.netloc,
                         quote(imgurl.path), imgurl.params,
                         imgurl.query, imgurl.fragment))
                            
    ressource = None

    # check URL

    try:
        ressource = urlopen(imgurl, timeout=TIMEOUT)
    except URLError as err:
        msg = "Cannot access ressource at url '{}': {}.".format(imgurl, err.reason) 
        return httpd.respond_with_user_error(1, msg)
    except timeout as e:
        msg = "Cannot access ressource at url '{}'. Timeout.".format(imgurl)
        return httpd.respond_with_user_error(22, msg)
    except ValueError as err:
        msg = "Cannot access ressource at url '{}'. Invalid URL.".format(imgurl, str(type(err))) 
        return httpd.respond_with_user_error(2, msg)
    except Exception as e:
        msg = "Cannot access ressource at url '{}'. Exception {} occured.".format(imgurl, str(type(e)))
        return httpd.respond_with_error(500, msg)
    
    resinfo = ressource.info()
    if 'Content-Length' in resinfo:
        # TODO catch non int content-length
        if int(resinfo['Content-Length']) >= MAX_FILE_SIZE:
            msg = "Cannot download ressource at url '{}': File is too big (Max file size: {}kB).".format(imgurl, MAX_FILE_SIZE / 1024) 
            return httpd.respond_with_user_error(3, msg)


    # download file

    f = None
    try:
        f = open('/tmp/rasta_tmp','wb')
        f.write(ressource.read(MAX_FILE_SIZE))
    except Exception as e:
        msg = "Cannot download ressource at url '{}'. Exception {} occured.".format(imgurl, str(type(e)))
        return httpd.respond_with_error(500, msg)
    finally: 
        f.close()

    try:
        img = Image.open('/tmp/rasta_tmp')
        img.close()
    except IOError:
        msg = "Url does not point to a supported image file."
        return httpd.respond_with_user_error(4, msg)

    # making prediction

    try:
        pred,pcts = get_pred(model, '/tmp/rasta_tmp', IS_DECAF, K)
    except Exception as e:
        msg = "Cannot load image, please try another image."
        return httpd.respond_with_user_error(5, msg)
    
    pcts = [ str(i) for i in pcts ]
    resp = { 'pred' : pred, 'pcts' : pcts, 'k' : K }

    return httpd.respond(resp)

class Handler(http.server.BaseHTTPRequestHandler):

    query_dispatcher = { 'predict' : query_predict }

    def respond(self, data):
        data['error'] = 200
        datastr = json.dumps(data, ensure_ascii=False)
        self.send_response(data['error'])
        self.send_header("Content-type", 'text/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(datastr.encode('utf-8'))
        print("Response ({}): ".format(datastr).encode('utf-8'));
        return 200
	
    def respond_with_error(self, err, msg):
        print("Sending error: " + msg)
        datastr = json.dumps({ 'error' : err, 'error_msg' : msg }, ensure_ascii=False)
        self.send_response(err)
        self.send_header("Content-type", 'text/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(datastr.encode("utf-8"))
        print("Response (Error!: {}): {}".format(err, datastr.encode("utf-8")))
        return err

    def respond_with_user_error(self, user_err_code, user_err_msg):
        resp = {"user_error": user_err_code,
                "user_error_msg": user_err_msg }
        return self.respond(resp)

    def do_GET(self):
        req = urlparse(self.path)
        msg = "Ok"
        query = None

        try:
            query = parse_qs(req.query, strict_parsing = True)
        except ValueError as e:
            msg = "Error: cannot parse query: '" + req.query + "'"
            return self.respond_with_error(422, msg)

        if 'remote_addr' in query:
            print('Handling proxy query from ' + str(query['remote_addr']))

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

    sys.stderr = open('rasta.log', 'w')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
        sys.exit(0)


if __name__ == '__main__':
    main()