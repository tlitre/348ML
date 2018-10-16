class Node:
  def __init__(self):
    self.label = None
    self.children = {}
	# you may want to add additional fields here...
    self.decision_attribute = ""
    self.previous_decisions = []
    self.examples = []
    self.depth = 0
    self.decisionMade = ""
    self.value = None
