import fast

model = fast.train_unsupervised('data/corpus/R8.clean.txt')

model.save_model("model")

