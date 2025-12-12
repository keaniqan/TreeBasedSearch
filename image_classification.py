import tensorflow as tf

# Global variable to hold the loaded model
model = None
preprocess = None

def load_model(model_type: str):
    """
    Loads a Keras model from the specified file path and saves it to the global variable 'model'.
    Args:
        file_path (str): Path to the Keras model file
    """  
    global model, preprocess
    # Based on the models defined in constants.ENUM_AI_MODELS load the appropriate model
    if(model_type == "MobileNetV2"):
        file_path = "models\\best_finetuned_mobilenetv2.keras"
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif(model_type == "ResNet50"):#TOOD: TEMPORARARY TESTING, REPLACE WITH ACTUAL MODEL LATER
        file_path = "models\\resnet_finetuned.keras"
        preprocess = tf.keras.applications.resnet50.preprocess_input
    elif(model_type == "Xception"):
        file_path = "models\\xception_finetuned.keras"
        preprocess = tf.keras.applications.xception.preprocess_input
    model = tf.keras.models.load_model(file_path)
    print(f"✅ Model '{model_type}' loaded from {file_path}")

def clasify_accident(image_path:str)-> int:
    """Clasify the image passed
    Using a image recognition model clasify the severity of the car crash captured
    in the image. The accident severity is classified in 4 levels:
        0 - No Accident
        1 - Minor Accident
        2 - Moderate Accident
        3 - Severe Accident
    Args:
        image_path (string): Path to image
        debug (bool): If True, print debug information

    Returns:
        int: Severity of the accident (0-3)
        list: Raw prediction scores from the model
    """
    global model
    if model is None:
        raise ValueError("Model is not loaded. Please load a model before classification.")

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension at axis 0
    img_array = preprocess(img_array)

    predictions = model.predict(img_array)
    severity = tf.argmax(predictions[0]).numpy()
    return severity, predictions[0]

#main runner for command line testing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify accident severity from an image using a pre-trained model.")
    parser.add_argument("model_type", type=str,choices=["ResNet50", "MobileNetV2", "Exception"], help="Type of the model to use for classification")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    model_type = args.model_type
    image_path = args.image_path
    
    load_model(model_type)
    severity, predictions = clasify_accident(image_path)
    print(f"Accident Severity: {severity}")
    print(f"Predictions: {predictions}")