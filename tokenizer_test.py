from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained('tok')

print (tokenizer('hi'))
# 'input_ids': [0, 262:'hi', 2] 
# - The exact complete word is preferred

print (tokenizer('twitter hi'))
# 'input_ids': [0, 88:'t', 280:'wit', 279:'ter', 225:'', 262:'hi', 2]
# - a mixture of subwords and words are observed. in further review,
# we have 'Ġtwit' in dictinary, and 'Ġ' is a token meaning :
# "continued after another token".
# because we didn't use the word 'twitter' after another token, it
# prefers to not use 'Ġtwit'

print (tokenizer('test tokenizer'))
# 'input_ids': [0, 88:t, 269:es, 88:t, 288:Ġtoken, 275:iz, 268:er, 2]
# - another mixture of words and subwords, and the tokenizer still prefers
# complete words to smaller, sub-word chunks

print (tokenizer('unknown words'))
# 'input_ids': [0, 89, 82, 79, 82, 83, 91, 82, 225, 91, 290, 87, 2]
# - for unknown words, the tokenizer goes full sub-word mode and
# breaks them apart to their alphabets
