import spacy
import pytextrank

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank", last=True)

text = "After using LaTeX a lot, any other typeset mathematics just looks hideous."
#       After using LaTeX a lot, any other typeset mathematics just looks hideous.
doc = nlp(text)

for sent in doc._.textrank.summary(limit_phrases=1, limit_sentences=1):
    print(sent.text)