from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=['sample_text.txt'],
                vocab_size=30_522,
                min_frequency=1,
                special_tokens=
                ['<s>', '<pad>', '</s>', '<ukn>', '<mask>'
                 ])
tokenizer.save_model('tok')