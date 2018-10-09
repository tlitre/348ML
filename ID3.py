from node import Node
import math

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  t = Node()
  if examples == None:
  #TODO check examples classification
  else:
    info = info_gain(examples)
    best = min(info, key=info.get)
    t.decision_attribute = best
    for i in examples:
      i.


def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''


def info_gain(examples):
  attribute_prob = {}
  for key in examples[0]:
    if key != 'Class':
      attribute_prob[key] = 0
  for i in examples:
    c = i['Class']
    for key, value in i:
      if key != 'Class':
        if value == c:
          attribute_prob[key] += 1
  res = {}
  for att, value in attribute_prob:
    attribute_prob[att] = (value / len(examples))
    ent = -attribute_prob[att] * math.log(attribute_prob[att], 2)
    res[att] = attribute_prob[att] * ent
  return res
    
      
'''
def find_entropy(examples, probabilities):
  summer = {}
  for i in examples:
    if !summer[i['Class']]:
      summer[i['Class']] = 1
    else:
      summer[i['Class']] += 1
  res = 0
  for key in summer.keys():
    x = res[key]
    ent = x - (x / len(examples)) * math.log(x / len(examples), 2)
    res['key'] = ent
  ent
'''