
from node import Node
import math

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  ##print("Training")
  t = Node()
  t.value = ""
  #This will see if we are a recursive call
  if default != 0:
    if type(default) is Node:
      t = default
    elif type(default) is str:
      t.value = default
  #This is for if we are given a set of no examples at all
  if examples == None:
    return t
  #TODO check examples classification
  else:
    #print("Running to check classification special examples")
    #This will check if the classification of the examples are all identical, or if all attr values are equal
    sameClassification = 1
    sameAttributes = 1 
    attrVals = {}
    ##print("Assigning initial examples")
    for i in examples[0]:
      attrVals[i] = examples[0][i]
    ##print("Going through each example and comparing to example 1 values")
    for i in examples:
      for j in i.keys():        
        if j == 'Class' and sameClassification:
          if i[j] != attrVals[j]:
            ##print("Not the same Classification")
            sameClassification = 0
        else:
          if i[j] != attrVals[j] and sameAttributes:
            ##print("Not same attributes")
            sameAttributes = 0

    ##print("Now assigning MODE for Node for pruning and indecision")
    #This will find the MODE answer for every node for use when we can't make a decision or for pruning
    classCount = {}
    for i in examples:
      if i['Class'] in classCount:
        classCount[i['Class']] += 1
      else:
        classCount[i['Class']] = 1
    t.value = max(classCount, key=classCount.get)
    
    if len(t.previous_decisions) == len(examples[0]) - 1:
      t.decisionMade = 'Y'
      return t

    #The examples all have the same classification
    if sameClassification:
      ##print("We all have the same classification")
      #We set the current node we have to just be any of the 'Class' values
      t.decisionMade = 'Y'
      t.value = examples[0]['Class']
      return t

    #The examples are all the same input vals, we return the mode of Class
    elif sameAttributes:
      ##print("We all have the same attributes")
      t.decisionMade = 'Y'
      return t

    #The examples are not a special case and we must compute IG and make a decision for the tree
    else:
      ##print("Calculating Info Gain")
      info = info_gain(examples)
      ##print("Grabbing best attribute from info gain: ")
      unused_attributes = {}
      for i in info.keys():
        if not(i in t.previous_decisions):
          unused_attributes[i] = info[i]
      best = min(unused_attributes, key=info.get)
      ##print(best)
      t.decision_attribute = best
      t.previous_decisions.append(best)
      #Sort examples based on best attribute 
      ##print("Sorting examples to the new children")
      ##print(best)
      ##print(examples)
      for i in examples:
        if i[best] in t.children.keys():
          ##print("Found child node")
          t.children[i[best]].examples.append(i)
        else:
          ##print("Making new child for node")
          newNode = Node()
          newNode.label = i[best]
          newNode.depth = t.depth + 1
          newNode.previous_decisions = t.previous_decisions
          newExamples = []
          newExamples.append(i)
          newNode.examples = newExamples
          t.children[i[best]] = newNode
            
      ##print(t.children)
      ##print("Running ID3 on children")
      #Run recursion on all the children
      for i in t.children:
        ##print(i)
        ##print(t.children[i].examples) 
        i = ID3(t.children[i].examples,t.children[i])

        #i = ID3(t.children[i].examples, 0)
      return t

def prune(node, examples):
  #print("pruning")
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  #The method will be to work down a branch at a time and test taking the mode of each node instead of navigating further
  #if it improves accuracy, then we will kill all the children (lol) and set decisionMade to "Y"
  baseAccuracy = test(node, examples)
  for i in node.children:
    if node.children[i].decisionMade != 'Y':
      node.children[i].decisionMade = 'Y'
      newAccuracy = test(node, examples)
      if newAccuracy >= baseAccuracy:
        #kill children's children
        node.children[i].children = {}
      else:
        node.children[i].decisionMade = ''
        #potentially prune the childrens children
        i = prune(node.children[i], examples)
  return node

def test(node, examples):
  #print("Testing")
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  totalExamples = len(examples)
  correct = 0.0
  for i in examples:
    result = evaluate(node, i)
    if result == i['Class']:
      correct += 1.0
  return correct / totalExamples

def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  #Check if we have arrived at a solution Node
  if node.decisionMade == 'Y':
    return node.value

  direction = example[node.decision_attribute]
  if direction in node.children: 
    newNode = node.children[direction]
  else:
    return node.value

  #Make sure that the child exists
  if(newNode != None):
    return evaluate(newNode, example)
  #if it doesn't, we will take the MODE of the examples it trained on
  else: 
    return node.value
  

def info_gain(examples):
  attribute_prob = {}
  entropies = {}
  split_examples = {}
  split_examples_res = {}
  res = {}

  for key in examples[0]:  
    if key != 'Class':
      attribute_prob[key] = 0

  for i in examples:
    c = i['Class'] 
    for key, value in i.items(): 
      if key != 'Class' and value == c:
        attribute_prob[key] += 1
  for i in attribute_prob:
    split_examples.clear()
    for j in examples:
      if j[i] in split_examples:
        split_examples[j[i]].append(j)
      else:
        split_examples[j[i]] = []
        split_examples[j[i]].append(j)
        for j in split_examples:
          split_examples_res[j] = find_entropy(split_examples[j], i)
    entropies[i] = 0
    for j in split_examples_res:
      entropies[i] += split_examples_res[j]

  for att in attribute_prob:
    res[att] = attribute_prob[att] * entropies[att]

  
  return res
    
      

def find_entropy(examples, attr):
  ent = 0.0
  invEnt = 0.0
  for i in examples:
    if i[attr] == i['Class']: 
      ent += 1
  ent = ent / len(examples)
  if ent != 0:
    invEnt = 1 / ent
    ent = -ent*math.log(ent,2)
    ent += -invEnt*math.log(invEnt,2)
  else:
    ent = 0
  return ent

