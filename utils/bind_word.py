def bind_word(sentences, w_model):
    for sentence_index in range(0, len(sentences)):
        sentences[sentence_index] = list(filter(
            lambda word: word in w_model.wv.vocab,
            sentences[sentence_index]
        ))

        for word_index in range(0, len(sentences[sentence_index])):
            original_text = sentences[sentence_index][word_index]

            sentences[sentence_index][word_index] = w_model[original_text]

    return sentences
