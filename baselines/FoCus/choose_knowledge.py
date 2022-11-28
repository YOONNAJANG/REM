import torch
import numpy as np

def choose_knowledge_with_tfidf(knowledge, question, table):

    for i, paragraph in enumerate(knowledge):
        table.add_document(i, paragraph)
    results = table.similarities(question)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    result_idx = [i[0] for i in results[:5]]
    chosen_knowledge = [knowledge[ri] for ri in result_idx]
    return chosen_knowledge

def choose_knowledge_sentence_with_tfidf(knowledge, question, sentence_knowledge_vector_dict, landmark_link, model_key, table):
    # sent_list = []
    # for paragraph in knowledge:
    #     sentences = sent_tokenize(tokenizer.decode(paragraph))
    #     sent_list += sentences

    for i, sent in enumerate(sentence_knowledge_vector_dict[landmark_link][model_key]):
        table.add_document(i, sent)
    results = table.similarities(question)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    result_idx = [i[0] for i in results[:5]]
    chosen_knowledge = [sentence_knowledge_vector_dict[landmark_link][model_key][ri] for ri in result_idx]
    return chosen_knowledge

def choose_knowledge_dpr(landmark_link, question, tokenizer, table):
    results = table.similarities(landmark_link, question, tokenizer)
    chosen_knowledge = results[:10]
    return chosen_knowledge

def choose_knowledge_trained_dpr(landmark_link, question, sentence_knowledge_vector_dict, model_key, tokenizer, table):
    top_k = 5

    table.eval()
    with torch.no_grad():
        outputs = table(landmark_link=landmark_link,
                        question=tokenizer.decode(question))

    enc_knowledge = sentence_knowledge_vector_dict[landmark_link][model_key]
    sort_rl = np.argpartition(outputs[2], -top_k)
    sort_rl = sort_rl[::-1].tolist()
    sorted_knowledge = [enc_knowledge[x] for x in sort_rl[:top_k]]
    return sorted_knowledge


def choose_knowledge_with_bm25(knowledge, question, BM25Okapi):
    bm25 = BM25Okapi(knowledge)
    chosen_knowledge = bm25.get_top_n(question, knowledge, n=5)
    return chosen_knowledge

def choose_knowledge_sentence_with_bm25(knowledge, question, sentence_knowledge_vector_dict, landmark_link, model_key, BM25Okapi):
    bm25 = BM25Okapi(sentence_knowledge_vector_dict[landmark_link][model_key])
    chosen_knowledge = bm25.get_top_n(question, sentence_knowledge_vector_dict[landmark_link][model_key], n=5)
    return chosen_knowledge

