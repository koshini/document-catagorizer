from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    file_path = 'document.txt'

    tfidf_representation = calculateTFIDF(file_path)
    # Print the matrix
    for list in tfidf_representation.toarray():
        print (list.tolist())



def calculateTFIDF(file_path):
    corpus = []
    with open(file_path,'r') as file:
        line = file.readline()
        cnt = 1
        while line:
            corpus.append(line)
            line = file.readline()
            cnt += 1

    tokenize = lambda doc: doc.lower().split(" ")
    tf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)

    tfidf_representation = tf.fit_transform(corpus)

    return tfidf_representation


if __name__ == "__main__":
    main()