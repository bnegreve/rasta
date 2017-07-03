# json based shell for Rasta as a service
# Example query:
# { "type":"predict", "url":"http://www.lamsade.dauphine.fr/~bnegrevergne/webpage/software/rasta/rasta-project/data/wikipaintings_10/wikipaintings_test/Post-Impressionism/adam-baltatu_olt-river-at-cozia.jpg"}

from sys import stdin
from urllib.request import urlretrieve
from urllib.error import URLError
from evaluation import get_pred, init
import json

MODEL_PATH='../savings/resnet_2017_6_29-18:42:50/model.h5'
IS_DECAF = False
K = 3

def main():
    model = init(MODEL_PATH, IS_DECAF)
    for line in stdin:
        resp = ''
        print(process_query(model, line))    

    
def respond(content):
    content['error'] = 200
    return json.dumps(content).encode("utf-8")


def respond_with_error(err, msg):
    return json.dumps({ 'error' : err, 'error_msg' : msg }).encode("utf-8")


def query_predict(model, query):

    result = {}

    if not 'url' in query:
        msg = "Error: missing url argument in predict query"
        return respond_with_error(422, msg)

    url = query['url']
    ressource = None

    try:
        print("Downloading URL " + url)
        (filename, ressource) = urlretrieve(url, '/tmp/rasta_tmp')
        #print("FILE " + str(ressource))
    except URLError as err:
        resp = {"error": 200,
                "user_error_msg": "Cannot download ressource at url '{}': {}".format(url, err.reason) }
        return respond(resp)
    except ValueError as err:
        resp = {"error": 500,
                "user_error_msg": "Cannot download ressource at url '{}'. Invalid URL.".format(url, str(type(err))) }
        return respond(resp)
    except Exception as e:
        resp = {"user_error": 500,
                "user_error_msg": "Cannot download ressource at url '{}'. Exception {} occured.".format(url, str(type(e))) }
        return respond(resp)

    pred,pcts = get_pred(model, '/tmp/rasta_tmp', IS_DECAF, K)

    pcts = [ str(i) for i in pcts ]
    resp = { 'pred' : pred, 'pcts' : pcts, 'k' : K }

    return respond(resp)


query_dispatcher = { 'predict' : query_predict }

def process_query(model, data):
    print("Processing query string: '{}'".format(data.rstrip()))
    
    query = None

    try:
        query = json.loads(data)
    except json.JSONDecodeError as e:
        return respond_with_error(400, "Error: invalid request format (should be json)")

    if not 'type' in query:
        return respond_with_error(400, "Error: no type attribute specified in query")

    qtype = query['type']
    if not qtype in query_dispatcher:
        return response_with_error(400, "Error: '{}' is not a valid query type ".format(qtype))
    
    return query_dispatcher[qtype](model, query)

if __name__ == '__main__':
    main()

