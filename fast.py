import fasttext

model = fasttext.train_unsupervised('data/corpus/R8.clean.txt')

model.save_model("model")

