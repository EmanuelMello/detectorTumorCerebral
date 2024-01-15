import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten,BatchNormalization
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

dim = 64

datagen = ImageDataGenerator(rescale = 1./255)

datatrain = datagen.flow_from_directory('D:\Dowloads\cnn-novo\cnn\Training', 
                                        target_size = (dim,dim), 
                                        batch_size = 2, 
                                        class_mode='categorical',
                                        shuffle = True)

datatest = datagen.flow_from_directory('D:\Dowloads\cnn-novo\cnn\Testing', 
                                        target_size = (dim,dim), 
                                        batch_size = 1, 
                                        class_mode='categorical')

earlyStopping = callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', patience=40, restore_best_weights=True, verbose=1)

callbacks_list = [earlyStopping]

with tf.device('CPU'):
    rede = Sequential()
    
    rede.add(Conv2D(32, (3,3),input_shape =(dim,dim,3),activation='relu'))
    rede.add(BatchNormalization())
    rede.add(MaxPooling2D(pool_size = (2,2)))
        
    rede.add(Conv2D(32, (3,3),input_shape =(dim,dim,3),activation='relu'))
    rede.add(BatchNormalization())
    rede.add(MaxPooling2D(pool_size = (2,2)))
    
    rede.add(Conv2D(32, (3,3),input_shape =(dim,dim,3),activation='relu'))
    rede.add(BatchNormalization())
    rede.add(MaxPooling2D(pool_size = (2,2)))
        
    rede.add(Flatten())
    
    rede.add(Dense(units=300, activation='relu'))
    #rede.add(Dropout(0.05))
    rede.add(Dense(units=300, activation='relu'))
    #rede.add(Dropout(0.05))
    rede.add(Dense(units=300, activation='relu'))
    #rede.add(Dropout(0.05))
    rede.add(Dense(units=300, activation='relu'))
    #rede.add(Dropout(0.05))
    rede.add(Dense(units=300, activation='relu'))
    #rede.add(Dropout(0.05))
    
    rede.add(Dense(units=3, activation='softmax'))
    
    rede.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    rede.fit(datatrain,validation_data=datatest,epochs=1000,callbacks=callbacks_list)
    
    test_loss, test_accuracy = rede.evaluate(datatest, verbose=1)
    print(f'Test Accuracy: {test_accuracy * 100:.6f}%')
    
    rede.save('modelTreinado.h5')
    

