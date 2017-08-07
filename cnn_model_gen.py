# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

class my_CNN_Model:
    def __init__(self, model_name="Resnet50"):
        self.model_name = model_name
        filename = 'bottleneck_features/Dog'+model_name+"Data.npz"
        try:
            self.bottleneck_features = np.load(filename)
        except:
            print("Can't find bottleneck features for "+model_name)
        self.train = self.bottleneck_features['train']
        self.valid = self.bottleneck_features['valid']
        self.test = self.bottleneck_features['test']
        self.save_weights = 'saved_models/weights.best.'+model_name+'.hdf5'
        
    def create_model(self):
        self.model = Sequential()
        self.model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
        self.model.add(Dense(133, activation='softmax'))
        self.model.summary()
        
    def compile_model(self):
        self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        
    def train_model(self):
        checkpointer = ModelCheckpoint(filepath=self.save_weights,verbose=1, save_best_only=True)
        self.model.fit(self.train, train_targets, validation_data=(self.valid, valid_targets), epochs=20,
                       batch_size=20, callbacks=[checkpointer], verbose=1)
    
        self.model.load_weights(self.save_weights)
        
    def test_model(self):
        predictions = [np.argmax(self.model.predict(np.expand_dims(feature, axis=0))) for feature in self.test]
        test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
        print('Test accuracy: %.4f%%' % test_accuracy)
        
    def predict_breed(self, img_path):
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = Resnet50_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        winners =dict(zip(dog_names, predicted_vector[0]))
        top5 = list(sorted(dog_names,key=lambda x: -winners[x]))[:5]
        
        return [(breed, winners[breed]) for breed in top5]
    
    def final_predictor(self,img_path):
        human = face_detector(img_path)
        dog = dog_detector(img_path)
        
        if not human and not dog:
            return "I don't see a human or a dog in that photo!  Try a different one."
        breeds = self.predict_breed(img_path)
        
        if human:
            salutation = "Hello, human!"
        else:
            salutation = "What a cute doge!"
            
        print(salutation)
        plt.imshow(img_path)
        print("You look most like a...")
        for entry in breeds:
            print("{0:.2%} {1}".format(breeds[1], breeds[0]))
    
    