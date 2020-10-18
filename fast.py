import fasttext

model = fasttext.train_unsupervised('data/corpus/R8.clean.txt', dim=300)

model.save_model("model")

# model = fasttext.load_model("model")
x = model.get_word_vector("the")

