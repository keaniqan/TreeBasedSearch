import tensorflow as tf

# Global variable to hold the loaded model
model = None


def load_model(file_path: str):
    """
    Loads a Keras model from the specified file path and saves it to the global variable 'model'.
    Args:
        file_path (str): Path to the Keras model file
    """  
    global model
    model = tf.keras.models.load_model(file_path)

def clasify_accident(image_path:str, debug:bool=False)-> int:
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
    """
    global model
    if model is None:
        raise ValueError("Model is not loaded. Please load a model before classification.")

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension at axis 0
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    if debug:
        print(f"Predictions: {predictions}")
    severity = tf.argmax(predictions[0]).numpy()
    return severity

#main runner for command line testing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify accident severity from an image using a pre-trained model.")
    parser.add_argument("model_path", type=str, help="Path to the Keras model file")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()


    model_path = args.model_path
    image_path = args.image_path
    debug = args.debug
    
    load_model(model_path)
    severity = clasify_accident(image_path, debug=debug)
    print(f"Accident Severity: {severity}")