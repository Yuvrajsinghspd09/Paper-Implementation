'''
Build GPT2Classifier:
   * Combine GPT2FeatureExtractor and MLPClassifier
   * Implement methods for feature extraction, training, and classification
'''
class GPT2Classifier:
  def __init__(self,config):
    self.feature_extractor = GPT2FeatureExtractor()
    self.classifier = MLPClassifier(config)


  def train(self,texts,labels):
    features = self.feature_extractor.extract_features(texts)
    self.classifier.train(features,labels)

  def predict(self,texts):
    features = self.feature_extractor.extract_features(texts)
    return self.classifier.predict(features)
    
