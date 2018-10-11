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
  #This will see if we are a recursive call
  if default != 0:
    t = default
  #This is for if we are given a set of no examples at all
  if examples == None:
    return t
  #TODO check examples classification
  else:

    #This will check if the classification of the examples are all identical, or if all attr values are equal
    sameClassification = 1
    sameAttributes = 1 
    attrVals = {}
    for i in examples[0]:
      attrVals[i] = examples[0][i];
    for i in examples:
      for j in i:
        if j == "Class":
          if i[j] != attrVals[j]:
            sameClassification = 0
        else:
          if i[j] != attrVals[j]:
            sameAttributes = 0
    #This will find the MODE answer for every node for use when we can't make a decision or for pruning
    classCount = {}
    for i in examples:
      i[examples["Class"]] += 1
    t.value = max(classCount, key=classCount.get)

    #The examples all have the same classification
    if sameClassification:
      #We set the current node we have to just be any of the 'Class' values
      t.decisionMade = "Y"
      t.value = examples[0]['Class']
      return t

    #The examples are all the same input vals, we return the mode of Class
    elif sameAttributes:
      t.decisionMade = "Y"
      return t

    #The examples are not a special case and we must compute IG and make a decision for the tree
    else:
      info = info_gain(examples)
      best = min(info, key=info.get)
      t.decision_attribute = best
      exampleDict = {}
      #Sort examples based on best attribute 
      for i in examples:
        if i[best] in t.children:
          exampleDict[i[best]].append(i)
        else:
          newNode = Node()
          newNode.label = i[best]
          newNode.depth = t.depth + 1
          newExamples = []
          newExamples.append(i)
          exampleDict[i[best]] = newExamples
          t.children[i[best]] = newNode
      #Run recursion on all the children
      for i in t.children:
        i = ID3(exampleDict[i.label],i)

      return t

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  #The method will be to work down a branch at a time and test taking the mode of each node instead of navigating further
  #if it improves accuracy, then we will kill all the children (lol) and set decisionMade to "Y"
  baseAccuracy = test(node, examples)
  childrenToKill = []
  for i in node.children:
    if i.decisionMade != "Y"
      i.decisionMade = "Y"
      newAccuracy = test(node, examples)
      if newAccuracy >= baseAccuracy:
        #kill children's children
        i.children = {}
      else:
        i.decisionMade = ""
        #potentially prune the childrens children
        i = prune(i, examples)
  return node

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  totalExamples = len(examples)
  correct = 0
  for i in examples:
    result = evaluate(node, i)
    if result == i["Class"]:
      correct += 1
  return correct / totalExamples

def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  #Check if we have arrived at a solution Node
  if node.decisionMade == "Y":
    return node.value

  direction = example[node.decision_attribute]
  newNode = node.children[direction]

  #Make sure that the child exists
  if(newNode != None):
    return evaluate(newNode, example)
  #if it doesn't, we will take the MODE of the examples it trained on
  else: 
    return node.value
  

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
