import tensorflow as tf

# Global variable to hold the loaded model
model = None

def clasify_accident(image_path)-> int:
    """Clasify the image passed
    Using a image recognition model clasify the severity of the car crash captured
    in the image. The accident severity is classified in 4 levels:
        0 - No Accident
        1 - Minor Accident
        2 - Moderate Accident
        3 - Severe Accident
    Args:
        image_path (string): Path to image

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
    print(predictions)
    severity = tf.argmax(predictions[0]).numpy()
    return severity

def load_model(file_path):
    """
    Loads a Keras model from the specified file path and saves it to the global variable 'model'.
    Args:
        file_path (str): Path to the .keras model file
    """
    global model
    model = tf.keras.models.load_model(file_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
             loss=tf.keras.losses.CategoricalCrossentropy(),
             metrics=["accuracy"])

#main runner for command line testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python image_classification.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    load_model(model_path)
    severity = clasify_accident(image_path)
    print(f"Accident Severity: {severity}")