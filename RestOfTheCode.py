def load_config(file_path):
  with open(file_path,'r') as file:
    return yaml.safe_load(file)


def load_data(file_path):
  pass


def evaluate(model, texts, labels):
    predictions = model.predict(texts)
    accuracy = ((predictions > 0.5).float() == labels).float().mean()
    return accuracy.item()



# main script
def main():
  config = load_config('config.yaml')
  train_texts, train_labels = load_data('train_data.txt')
  test_texts,test_labels = load_data('test_data.txt')

  classifier = GPT2Classifier(config)
  classifier.train(train_texts,train_labels)

  accuracy = evaluate(classifier,test_texts,test_labels)
  print(f"test accuracy:{accuracy}")

# Inference
    new_text = "This is a new prompt to classify."
    prediction = classifier.predict([new_text])
    print(f"Prediction for '{new_text}': {prediction.item()}")

# 9. Testing
def run_tests():
    # Implement test cases here
    pass

if __name__ == "__main__":
    main()
    run_tests()

  
