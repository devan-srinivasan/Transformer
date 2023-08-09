"""
File to process and create training / testing data
"""
# parse tons of sentences
# per sentence of length n we can create sub-sentences / fragments
# experiment with much longer sentences versus one
# takes 8s maybe to process like 25000 tokens in a batch (one step). so to train for 2h = 900 steps only.
# need a niche amount of text with like 22 mil words ... yikes