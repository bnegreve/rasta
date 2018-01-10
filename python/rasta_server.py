# rasta_server.py
#
# before running, make sure your locale is set to some UTF8 locale
# e.g. with export LANG=C.UTF-8
#

import http.server
import socketserver
import sys
import subprocess
import json
from urllib.error import URLError
from urllib.request import urlopen
from urllib.parse import unquote, quote, urlparse, parse_qs, urlunparse
from PIL import Image
from socket import timeout
from evaluation import get_pred, init
from datetime import datetime
import pathlib


PORT = 4444
MODEL_PATH='models/best_top_3/model.h5'
IS_DECAF = False
K = 3
TIMEOUT=10
MAX_FILE_SIZE=10 * 1024 * 1024 # 10 MB
USER_AGENT_STRING="Mozilla/5.0 (X11; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0"


model = None
cache = {}

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

    # urlopen does not seem to accept utf-8 paths, so decode the ascii
    # encoded-url, ascii encode the path part and reform the url
    # see: https://stackoverflow.com/questions/11818362/how-to-deal-with-unicode-string-in-url-in-python3
    imgurl = unquote(query['url'][0])                       # decode ascii encoded url to utf-8
    imgurl = urlparse(imgurl)                               # break url appart
    imgurl = urlunparse((imgurl.scheme, imgurl.netloc,      # ascii encode path and rebuild url
                        quote(imgurl.path), imgurl.params,
                        imgurl.query, imgurl.fragment))


    ressource = None

    # check URL 

    global cache
    if imgurl in cache:
        resp = cache[imgurl]
        httpd.log_message('CACHE-HIT: %s %s', imgurl, str(resp))
        return httpd.respond(resp)

    try:
#        req = Request(imgurl, {}, {'User-agent' : USER_AGENT_STRING})
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

    # For some reason the predictions are not always utf-8
    # and that seem to depend on the locale encoding. (I'm not sure what's going on, really)
    # This will decode the prediction based on the locale encoding and remove broken chars 
    # if any. 
    pcts = [ str(i) for i in pcts ]
    pred = [ bytes(p, sys.getfilesystemencoding(), 'replace').decode('utf-8') for p in pred ]
    resp = { 'pred' : pred, 'pcts' : pcts, 'k' : K }

                      
    cache[imgurl] = resp
    
                      
    return httpd.respond(resp)

class Handler(http.server.BaseHTTPRequestHandler):

    query_dispatcher = { 'predict' : query_predict }


    def respond_raw(self, code, data):
        data['error'] = code
        datastr = json.dumps(data, ensure_ascii=False)
        self.send_response(data['error'])
        self.send_header("Content-type", 'text/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(datastr.encode('utf-8'))
#        print("Response ({}): ".format(datastr).encode('utf-8'));
        return code
	
    def respond(self, data):
        self.respond_raw(200, data)
        self.log_message('RESPONSE: %s', str(data))
        return 200

    def respond_with_error(self, err, msg):
        data = { 'error' : err, 'error_msg' : msg }
        self.respond_raw(err, data)
        self.log_error('ERROR %s: %s', str(err), msg) 
        return err

    def respond_with_user_error(self, user_err_code, user_err_msg):
        resp = {"user_error": user_err_code,
                "user_error_msg": user_err_msg }
        self.log_message('USER-ERROR %s: %s', str(user_err_code), user_err_msg) 
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

        sys.stderr.flush()
        return 0


def create_log_file(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
    return path + '/rasta_log_'+datetime.now().strftime('%F-%H-%M-%S')

def main():
    httpd = socketserver.TCPServer(("", PORT), Handler)

    global model
    model = init(MODEL_PATH, IS_DECAF) # todo outch

    print("serving at port", PORT)

    logfilename = create_log_file('./log')
    sys.stderr = open(logfilename, 'w')
    print("logs are in " + logfilename)


    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
        sys.exit(0)


if __name__ == '__main__':
    main()
