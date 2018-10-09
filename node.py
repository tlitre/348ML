class Node:
  def __init__(self):
    self.label = None
    self.children = {}
	# you may want to add additional fields here...
    self.decision_attribute = ""
    self.examples = ()
    self.depth = 0